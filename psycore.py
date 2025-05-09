from src.system_manager import LocalCredentials, ConfigManager, LoggerController
from src.data.s3_handler import S3Handler, S3Bucket
from src.kg import BERT_KG, LLM_KG
from src.llm import ModelCatalogue
from src.llm.wrappers import ChatModelWrapper
from src.vector_database import CLIPEmbedder, PineconeService, Embedder, VectorService
from src.preprocessing.file_preprocessor import FilePreprocessor
from src.main import PromptStage, Elaborator, RAGElaborator, UserPromptElaboration
from src.main import RAGStage, RAGChatStage
import argparse
import logging
import sys
import json
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
        self.logger.debug("Exiting init_s3")

    def init_config(self,config_path=None):
        if config_path is None:
            config_path = "config.yaml"
        config = ConfigManager(config_path)
        LoggerController.initialize(config.get_log_level())
        self.logger = LoggerController.get_logger()
        primaryModelType = config.get_model()
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

            elif self.graphModel == "bert":
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

    def preprocess(self):
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
        files = files[1:3]
        # Process the files
        self.file_preprocessor.process_files(files)
        self.logger.debug("Exiting preprocess")


    def process_prompt(self, base_prompt):
        self.logger.debug("Entering process_prompt")
        prompt_stage = PromptStage(None, self.prompt_style)
        rag_elaborator = RAGElaborator(self.elaborator_model)
        elaborated_prompt = rag_elaborator.elaborate(base_prompt)
        chosen_rag_prompt, elaborated = prompt_stage.decide_between_prompts(base_prompt, elaborated_prompt)
        rag_stage = RAGStage(self.vdb, 20)
        rag_results = rag_stage.get_rag_prompt(chosen_rag_prompt)
        rag_chat_results = self.rag_chat.chat(base_prompt, rag_results)
        print(rag_chat_results)
        self.logger.debug("Exiting process_prompt")


    def __init__(self, config_path=None):
        self.embedder = CLIPEmbedder()
        self.init_config(config_path)
        self.init_vector_database()
        self.init_s3()
        self.rag_chat = RAGChatStage(self.main_wrapper, self.s3_handler)
        self.logger.debug("Exiting __init__")

    def text_interface(self):
        self.logger.debug("Entering text_interface")
        while True:
            print("Type 'exit' to exit the program")
            prompt = input("Enter a prompt: ")
            if prompt == "exit":
                break
            self.process_prompt(prompt)
        self.logger.debug("Exiting text_interface")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Psycore CLI")
    parser.add_argument("--config", type=str, help="Path to the config file")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess the data")
    parser.add_argument("--proceed", action="store_true", help="If preprocessing, allows program to work as normal afterwards rather than only preprocessing")
    args = parser.parse_args()
    psycore = Psycore(args.config)
    if args.preprocess:
        psycore.preprocess()
        if not args.proceed:
            exit(0)
    else:
        print("No preprocessing requested. Continuing with Psycore.")
    psycore.text_interface()