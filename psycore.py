from src.system_manager import LocalCredentials, ConfigManager, LoggerController
from src.data.s3_handler import S3Handler, S3Bucket
from src.data.s3_quick_fetch import S3QuickFetch
from src.kg import BERT_KG, LLM_KG, dict_data_to_relations
from src.llm import ModelCatalogue, EmbeddingType
from src.llm.wrappers import ChatModelWrapper, EmbeddingWrapper
from src.vector_database import CLIPEmbedder, LangchainEmbedder, AWSEmbedder, PineconeService, Embedder, VectorService
from src.preprocessing.file_preprocessor import FilePreprocessor
from src.main import PromptStage, Elaborator, RAGElaborator, UserPromptElaboration
from src.main import RAGStage, RAGChatStage, IterativeStage
import argparse
from src.evaluation import BERTEvaluator, RougeEvaluator, GraphEvaluator
import logging
import sys
import json
from src.evaluation import GraphEvaluator, RougeEvaluator, BERTEvaluator
parser = argparse.ArgumentParser()

class Psycore:

    def init_s3(self):
        self.logger.debug("Entering init_s3")
        self.s3_creds = {
        "aws_iam": LocalCredentials.get_credential('AWS_IAM_KEY'),
        "region": LocalCredentials.get_credential('AWS_DEFAULT_REGION').secret_key,
        "buckets": {
            "documents": LocalCredentials.get_credential('S3_DOCUMENTS_BUCKET').secret_key,
            "text": LocalCredentials.get_credential('S3_TEXT_BUCKET').secret_key,
            "images": LocalCredentials.get_credential('S3_IMAGES_BUCKET').secret_key,
            "graphs": LocalCredentials.get_credential('S3_GRAPHS_BUCKET').secret_key
            }
        }
        self.s3_handler = S3Handler(self.s3_creds)
        self.s3_quick_fetch = S3QuickFetch(self.s3_handler)
        self.logger.debug("Exiting init_s3")

    def init_config(self,config_path=None):
        if config_path is None:
            config_path = "config.yaml"
        config = ConfigManager(config_path)
        LoggerController.initialize(config.get_log_level())
        self.logger = LoggerController.get_logger()
        if config.is_document_range_enabled():
            self.document_ids = config.get_document_ids()
        else:
            self.document_ids = None
        self.loop_retries = config.get_iteration_loop_retries()
        self.iterator_pass_threshold = config.get_iteration_pass_threshold()
        self.rag_text_similarity_threshold = config.get_rag_text_similarity_threshold()
        primaryModelType = config.get_model()
        if config.get_embedding_method() == "langchain":
            try:
                modelType = ModelCatalogue.get_MEmbeddings()[config.get_embedding_model()]
                self.embedding_wrapper = EmbeddingWrapper(modelType)
                self.embedder = LangchainEmbedder(self.embedding_wrapper)
            except KeyError:
                raise ValueError(f"Embedding model type '{config.get_embedding_model()}' is not recognized in the ModelCatalogue as a multimodal embedding. \nOptions are {list(ModelCatalogue.get_MEmbeddings().keys())}")
        elif config.get_embedding_method() == "aws":
            self.embedder = AWSEmbedder(config.get_embedding_model())
        elif config.get_embedding_method() == "clip":
            self.embedder = CLIPEmbedder()
        try:
            modelType = ModelCatalogue.get_MLLMs()[primaryModelType]
            self.main_wrapper = ChatModelWrapper(modelType)
        except KeyError:
            raise ValueError(f"Model type '{primaryModelType}' is not recognized in the ModelCatalogue as an MLLM. \nOptions are {list(ModelCatalogue.get_MLLMs().keys())}")
        if config.get_text_summariser_model() is not None:
            modelType = ModelCatalogue.get_MLLMs()[config.get_text_summariser_model()]
            self.text_summariser = ChatModelWrapper(modelType)
        else:
            self.text_summariser = self.main_wrapper
        if config.get_elaborator_model() is not None:
            modelType = ModelCatalogue.get_MLLMs()[config.get_elaborator_model()]
            self.elaborator_model = ChatModelWrapper(modelType)
        else:
            self.elaborator_model = self.main_wrapper
        self.allow_mllm_images = config.allow_images()
        if config.is_graph_verification_enabled() == True:
            graphModel = config.get_graph_method()
            if graphModel == "llm":
                graphModelName = config.get_graph_llm_model()
                try:
                    modelType = ModelCatalogue.get_models_with_json_schema()[graphModelName]
                    wrapper = ChatModelWrapper(modelType)
                    self.graphModel = LLM_KG(wrapper, self.embedder)
                except KeyError:
                    raise ValueError(f"Graph model type '{graphModelName}' is not recognized in the ModelCatalogue as with json schema encoding.\n Options are {list(ModelCatalogue.get_models_with_json_schema().keys())}")

            elif graphModel == "bert":
                self.graphModel = BERT_KG()
        else:
            self.graphModel = None
        self.prompt_style = config.get_prompt_mode()
        self.logger.debug("Exiting init_config")

    def init_vector_database(self):
        self.logger.debug("Entering init_vector_database")
        self.vdb = PineconeService(self.embedder, {
            "index_name": LocalCredentials.get_credential('PINECONE_INDEX').secret_key,
            "api_key": LocalCredentials.get_credential('PINECONE_API_KEY').secret_key,
            "aws_region": LocalCredentials.get_credential('PINECONE_REGION').secret_key
        })
        self.logger.debug("Exiting init_vector_database")

    def preprocess(self, skip_confirmation=False):
        if not skip_confirmation:
            confirmation = input("Are you sure you want to preprocess the data? This will delete all existing data from the VDB and S3 buckets. (y/n): ")
            if confirmation != "y":
                print("Preprocessing cancelled.")
                return
        self.logger.debug("Entering preprocess")
        # Clean the VDB
        self.vdb.reset_data()
        # Clean the S3 Buckets
        self.s3_handler.reset_buckets()
        # Create the file preprocessor
        self.file_preprocessor = FilePreprocessor(self.s3_handler, self.vdb, self.embedder,self.text_summariser, self.graphModel)
        # Get all files from the Documents bucket
        files = self.s3_handler.list_base_directory_files(S3Bucket.DOCUMENTS) 
        # Limit to first 2 files for testing
        if self.document_ids is not None and len(self.document_ids) > 0:
            files = [files[i] for i in self.document_ids]
        # Process the files
        self.file_preprocessor.process_files(files)
        self.logger.debug("Exiting preprocess")


    def process_prompt(self, base_prompt, rag_elaborator : Elaborator = None):
        self.logger.debug("Entering process_prompt")
        prompt_stage = PromptStage(None, self.prompt_style)
        elaborated_prompt = rag_elaborator.elaborate(base_prompt)
        chosen_rag_prompt, elaborated = prompt_stage.decide_between_prompts(base_prompt, elaborated_prompt)
        rag_stage = RAGStage(self.vdb, 5)
        rag_results = rag_stage.get_rag_prompt_filtered(chosen_rag_prompt, self.rag_text_similarity_threshold)
        rag_chat_results = self.rag_chat.chat(base_prompt, rag_results)
        rag_elaborator.queue_history(rag_chat_results.content)
        print(f"Output:\n{rag_chat_results.content}\nSource:\n{[(result['document_path'], result['vector_id'], result['score']) for result in rag_results]}\nRAG Prompt:\n{chosen_rag_prompt}")
        self.logger.debug("Exiting process_prompt")

    def evaluate_prompt(self, base_prompt) -> dict:
        logger = LoggerController.get_logger()
        prompt_stage = PromptStage(None, self.prompt_style)
        elaborator = RAGElaborator(self.elaborator_model)
        elaborated_prompt = elaborator.elaborate(base_prompt)
        chosen_prompt, elaborated = prompt_stage.decide_between_prompts(base_prompt, elaborated_prompt)
        rag_stage = RAGStage(self.vdb, 10)
        rag_results = rag_stage.get_rag_prompt_filtered(chosen_prompt, self.rag_text_similarity_threshold)
        rag_chat_results = self.rag_chat.chat(base_prompt, rag_results)
        
        logger.info(rag_results)
        logger.info("Evaluating RAG results")

        iterative_stage = IterativeStage(self.s3_quick_fetch, self.graphModel, self.iterator_pass_threshold,rag_results)
        stage_results = iterative_stage.decision_maker(rag_results,rag_chat_results)
        retry_count = 0
        while len(stage_results[1]) > 0 and retry_count < self.loop_retries:
            missing_relations = stage_results[1]
            string_relations = [ str(relation) for relation in missing_relations]
            rag_chat_results = self.rag_chat.chat(base_prompt + ", bear in mind: " + ", ".join(string_relations), rag_results)
            stage_results = iterative_stage.decision_maker(rag_results,rag_chat_results)
            retry_count += 1
        print(stage_results)
        (threshold, valid_relations, missing_relations) = stage_results
        evaluators = [
            GraphEvaluator(iterative_stage, self.graphModel),
            BERTEvaluator(iterative_stage),
            RougeEvaluator(iterative_stage)
        ]
        for i in range(len(rag_results)):
            logger.info(f"Evaluating RAG result {i}")
            for evaluator in evaluators:
                logger.info(f"Evaluating with {evaluator.__class__.__name__}")
                rag_results[i] = evaluator.evaluate_rag_result(rag_chat_results.content, rag_results[i])
        logger.info("Finished evaluating RAG results")
        return rag_results


    def __init__(self, config_path=None):

        self.init_config(config_path)
        self.init_vector_database()
        self.init_s3()
        self.rag_chat = RAGChatStage(self.main_wrapper, self.s3_handler)
        self.logger.debug("Exiting __init__")

    def text_interface(self):
        self.logger.debug("Entering text_interface")
        elaborator = RAGElaborator(self.elaborator_model)
        while True:
            print("Type 'exit' to exit the program")
            prompt = input("Enter a prompt: ")
            if prompt == "exit":
                break
            self.process_prompt(prompt, elaborator)
        self.logger.debug("Exiting text_interface")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Psycore CLI")
    parser.add_argument("--config", type=str, help="Path to the config file")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess the data")
    parser.add_argument("--proceed", action="store_true", help="If preprocessing, allows program to work as normal afterwards rather than only preprocessing")
    parser.add_argument("--skip-confirmation", action="store_true", help="Skip confirmation prompts during preprocessing")
    args = parser.parse_args()
    psycore = Psycore(args.config)
    if args.preprocess:
        psycore.preprocess(skip_confirmation=args.skip_confirmation)
        if not args.proceed:
            exit(0)
    else:
        print("No preprocessing requested. Continuing with Psycore.")
    psycore.text_interface()