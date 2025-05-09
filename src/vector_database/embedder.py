from abc import ABC, abstractmethod

from langchain_text_splitters import TokenTextSplitter
from numpy import ndarray
class Embedder(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def embed(self, text: str) -> list:
        """Embed the given text into a vector representation."""
        pass

    def chunk_text(self,chunk_size=57, chunk_overlap=20):
        """Chunk the text into smaller pieces for embedding."""
        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(self.text)
        pass

    @abstractmethod
    def chunk_to_embedding(chunk_data : str) -> ndarray:
        """Convert chunk data to embedding."""
        pass


    @abstractmethod
    def embed_batch(self, texts: list) -> list:
        """Embed a batch of texts into vector representations."""
        pass