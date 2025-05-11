from numpy import ndarray
from transformers import CLIPProcessor, CLIPModel
from typing import BinaryIO, Union
from PIL import Image
from .embedder import Embedder
class CLIPEmbedder(Embedder):
    BASE_MODEL = "openai/clip-vit-base-patch32"
    PROCESSOR_MODEL = "openai/clip-vit-base-patch32"

    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained(self.BASE_MODEL)
        self.processor = CLIPProcessor.from_pretrained(self.PROCESSOR_MODEL)
        self.max_clip_length = 77

    
    def text_to_embedding(self,text : str):
        """Convert chunk data to embedding."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=self.max_clip_length)
        embedding = self.model.get_text_features(**inputs)
        embedding = embedding[0] / embedding.norm()
        embedding = embedding.detach().numpy()
        return embedding
    
    def image_to_embedding(self, image: Union[Image.Image, BinaryIO]):
        """Convert image data to embedding."""
        inputs = self.processor(images=image, return_tensors="pt", padding=True, truncation=True, max_length=self.max_clip_length)
        embedding = self.model.get_image_features(**inputs)
        embedding = embedding[0] / embedding.norm()
        embedding = embedding.detach().numpy()
        return embedding
    
