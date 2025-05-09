from src.vector_database.embedder import Embedder
from abc import ABC, abstractmethod
class VectorService(ABC):
    @abstractmethod
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        pass

    @abstractmethod
    def add_data(self, data: dict):
        """Add data to the vector database."""
        pass

    @abstractmethod
    def get_data(self, query: str) -> dict:
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