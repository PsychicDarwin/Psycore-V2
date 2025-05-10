from numpy import ndarray
from src.llm.wrappers import EmbeddingWrapper
from typing import BinaryIO, Union
from PIL import Image
from .embedder import Embedder
class LangchainEmbedder(Embedder):
    def __init__(self, embedding_wrapper: EmbeddingWrapper):
        super().__init__()
        self.embedding_wrapper = embedding_wrapper

    def text_to_embedding(self, text: str) -> ndarray:
        return self.embedding_wrapper.embed_query(text)

    def image_to_embedding(self, image: Union[Image.Image, BinaryIO]) -> ndarray:
        return self.embedding_wrapper.embed_image(image)
