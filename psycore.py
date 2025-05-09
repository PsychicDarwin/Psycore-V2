from src.system_manager.LocalCredentials import LocalCredentials
from src.system_manager.ConfigManager import ConfigManager
from src.data.s3_handler import S3Handler
from src.kg.bert import BERT_KG
from src.kg.llm import LLM_KG
from src.llm.model_catalogue import ModelCatalogue
from src.llm.wrappers import ChatModelWrapper

import argparse
parser = argparse.ArgumentParser()

class Psycore:

    def init_s3(self):
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

    def init_config(self,config_path=None):
        if config_path is None:
            config_path = "config.yaml"
        config = ConfigManager(config_path)
        primaryModelType = config.get_model()
        try:
            modelType = ModelCatalogue.get_MLLMs()[primaryModelType]
            self.main_wrapper = ChatModelWrapper(modelType)
        except KeyError:
            raise ValueError(f"Model type '{primaryModelType}' is not recognized in the ModelCatalogue as an MLLM. \nOptions are {list(ModelCatalogue.get_MLLMs().keys())}")
        
        self.allow_mllm_images = config.allow_images()
        if config.is_graph_verification_enabled() == True:
            graphModel = config.get_graph_method()
            if graphModel == "llm":
                graphModelName = config.get_graph_llm_model()
                try:
                    modelType = ModelCatalogue.get_models_with_json_schema()[graphModelName]
                    wrapper = ChatModelWrapper(modelType)
                    self.graphModel = LLM_KG(wrapper)
                except KeyError:
                    raise ValueError(f"Graph model type '{graphModelName}' is not recognized in the ModelCatalogue as with json schema encoding.\n Options are {list(ModelCatalogue.get_models_with_json_schema().keys())}")

            elif self.graphModel == "bert":
                self.graphModel = BERT_KG()
        else:
            self.graphModel = None
        self.prompt_style = config.get_prompt_mode()


    def preprocess(self):
        pass

    def __init__(self, config_path=None):
        self.init_config(config_path)
        self.init_s3()

    def text_interface(self):
        print("Welcome to Psycore! \n")
        pass
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Psycore CLI")
    parser.add_argument("--config", type=str, help="Path to the config file")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess the data")
    parser.add_argument("--continue", action="store_true", help="If preprocessing, allows program to work as normal afterwards rather than only preprocessing")
    args = parser.parse_args()
    psycore = Psycore(args.config)
    if args.preprocess:
        psycore.preprocess()
        if not args.continue_:
            exit(0)
    else:
        print("No preprocessing requested. Continuing with Psycore.")
    psycore.text_interface()