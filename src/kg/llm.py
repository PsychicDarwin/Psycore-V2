
import re
import json
import torch
from src.kg.graph_creator import GraphCreator
from src.llm.wrappers import ChatModelWrapper
from src.vector_database.embedder import Embedder
from langchain_experimental.graph_transformers import LLMGraphTransformer
class LLM_KG(GraphCreator):
    def __init__(self, model: ChatModelWrapper, embedder: Embedder):
        self.modelType = model
        self.embedder = embedder
        self.transformer = LLMGraphTransformer(
            llm = model.model,
            
        )
        

    # Due to chunk based processing approach, this will not work for a mediums of information
    # Like hypothetically a book of short stories, as they may contradict each other regarding graph relations
    def create_graph_json(self, text: str):
