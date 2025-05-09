from abc import ABC, abstractmethod
from typing import BinaryIO, Union
from PIL import Image
from langchain_text_splitters import TokenTextSplitter
from numpy import ndarray
class Embedder(ABC):
    @abstractmethod
    def __init__(self):
        self.chunk_size = 57
        self.chunk_overlap = 20
        pass

    def chunk_text(self,text) -> list:
        """Chunk the text into smaller pieces for embedding."""
        text_splitter = TokenTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = text_splitter.split_text(text)
        return chunks

    @abstractmethod
    def text_to_embedding(chunk_data : str) -> ndarray:
        """Convert chunk data to embedding."""
        pass

    @abstractmethod
    def image_to_embedding(image: Union[Image.Image, BinaryIO]) -> ndarray:
        """Convert image data to embedding."""
        pass