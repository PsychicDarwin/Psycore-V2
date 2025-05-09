from numpy import ndarray
from transformers import CLIPProcessor, CLIPModel
from typing import BinaryIO, Union
from PIL import Image
from src.vector_database.embedder import Embedder
class CLIPEmbedder(Embedder):
    BASE_MODEL = "openai/clip-vit-base-patch32"
    PROCESSOR_MODEL = "openai/clip-vit-base-patch32"

    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained(self.BASE_MODEL)
        self.processor = CLIPProcessor.from_pretrained(self.PROCESSOR_MODEL)

    
    def text_to_embedding(self,text : str) -> ndarray:
        """Convert chunk data to embedding."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=self.chunk_size)
        embedding = self.model.get_text_features(**inputs)
        embedding = embedding[0] / embedding.norm()
        embedding = embedding.numpy()
        return embedding
    
    def image_to_embedding(self, image: Union[Image.Image, BinaryIO]) -> ndarray:
        """Convert image data to embedding."""
        inputs = self.processor(images=image, return_tensors="pt", padding=True, truncation=True, max_length=self.chunk_size)
        embedding = self.model.get_image_features(**inputs)
        embedding = embedding[0] / embedding.norm()
        embedding = embedding.numpy()
        return embedding
    
