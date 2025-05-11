from numpy import ndarray
from src.vector_database.embedder import Embedder
from abc import ABC, abstractmethod
class VectorService(ABC):
    @abstractmethod
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        pass

    @abstractmethod
    def add_data(self, embedding : ndarray, data: dict):
        """Add data to the vector database."""
        pass

    @abstractmethod
    def batch_add_data(self, embeddings: list[ndarray], data_list: list[dict], batch_size: int = 100):
        """Add multiple data points to the vector database in batches.
        
        Args:
            embeddings (list[ndarray]): List of embeddings to add
            data_list (list[dict]): List of metadata dictionaries corresponding to each embedding
            batch_size (int, optional): Size of batches to process. Defaults to 100.
        """
        pass

    @abstractmethod
    def get_data(self, query: str, k: int = 5) -> dict:
        """Get data from the vector database."""
        pass

    @abstractmethod
    def delete_data(self, data_id: str):
        """Delete data from the vector database."""
        pass

    @abstractmethod
    def update_data(self, data_id: str, new_data: str):
        """Update data in the vector database."""
        pass

    @abstractmethod
    def reset_data(self):
        """Reset the vector database."""
        pass