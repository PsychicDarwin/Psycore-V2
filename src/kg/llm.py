
import re
import json
import torch
from src.kg.graph_creator import GraphCreator
from src.llm.wrappers import ChatModelWrapper
class LLM_KG(GraphCreator):
    def __init__(self, model: ChatModelWrapper):
        self.modelType = model
        pass

    def create_graph_json(self, text: str):
        pass