from src.system_manager.LocalCredentials import LocalCredentials
from src.system_manager.ConfigManager import ConfigManager
from src.data.s3_handler import S3Handler
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

    def init_config(self):
        config = ConfigManager("config.yaml")
        self.primary_llm = config.get_model()
        self.allow_mllm_images = config.allow_images()
        self.graph_verification = config.is_graph_verification_enabled()
        self.graph_method = config.get_graph_method()
        self.graph_model = config.get_graph_llm_model()
        self.prompt_style = config.get_prompt_mode()

    def __init__(self):
        self.init_config()
        self.init_s3()

        
        



        
if __name__ == "__main__":
    psycore = Psycore()