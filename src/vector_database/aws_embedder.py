from numpy import ndarray
from boto3 import client
from typing import BinaryIO, Union
from PIL import Image
import json
from .embedder import Embedder
from src.system_manager.LocalCredentials import LocalCredentials
import numpy as np
class AWSEmbedder(Embedder):
    MODEL_NAMES = [
        "amazon.titan-embed-image-v1"
    ]
    def __init__(self, model_name: str):
        super().__init__(90,30,1024)
        if model_name not in self.MODEL_NAMES:
            raise ValueError(f"Invalid model name: {model_name}, must be one of {self.MODEL_NAMES}")
        self.model_name = model_name
        self.client = client(
            service_name="bedrock-runtime",
            region_name=LocalCredentials.get_credential("AWS_DEFAULT_REGION").secret_key,
            aws_access_key_id=LocalCredentials.get_credential("AWS_IAM_KEY").user_key,
            aws_secret_access_key=LocalCredentials.get_credential("AWS_IAM_KEY").secret_key
        )
        
    def text_to_embedding(self, text: str) -> ndarray:
        response = self.client.invoke_model(
            modelId=self.model_name,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "inputText": text
            })
        )
        response_body = json.loads(response.get("body").read())
        if response_body.get("message") is not None:
            raise Exception(response_body.get("message"))
        embedding = response_body.get("embedding")
        return np.array(embedding)
    
    def image_to_embedding(self, image: Union[Image.Image, BinaryIO]) -> ndarray:
        image_base64 = self.image_to_base64(image)
        response = self.client.invoke_model(
            modelId=self.model_name,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "inputImage": image_base64
            })
        )
        response_body = json.loads(response.get("body").read())
        if response_body.get("message") is not None:
            raise Exception(response_body.get("message"))
        embedding = response_body.get("embedding")
        return np.array(embedding)