from abc import ABC, abstractmethod

class GraphCreator(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def create_graph_json(self, text: str):
        pass