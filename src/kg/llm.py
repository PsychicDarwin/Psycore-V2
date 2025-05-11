import re
import json
import torch
from .graph_creator import GraphCreator, GraphRelation
from src.llm.wrappers import ChatModelWrapper
from src.vector_database import Embedder
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document

class LLM_KG(GraphCreator):
    def __init__(self, model: ChatModelWrapper, embedder: Embedder):
        self.modelType = model
        self.embedder = embedder
        self.transformer = LLMGraphTransformer(
            llm = model.model,
        )
        
    # Due to chunk based processing approach, this will not work for a mediums of information
    # Like hypothetically a book of short stories, as they may contradict each other regarding graph relations
    def get_nodes_and_relations(self, text: str):
        splitText = self.embedder.chunk_text(text, chunk_size=2000, chunk_overlap=600)
        documents = [
            Document(page_content=chunk) for chunk in splitText
        ]
        graph_documents = self.transformer.convert_to_graph_documents(documents)
        nodes = []
        relationships = []
        for graph_document in graph_documents:
            nodes.extend(graph_document.nodes)
            relationships.extend(graph_document.relationships)
        return nodes, relationships


    def create_graph_relations(self, text: str):
        nodes, relationships = self.get_nodes_and_relations(text)
        triples = []
        for node in nodes:
            triplet = GraphRelation(node.id, node.type, "IS_A")
            triples.append(triplet)
        for relationship in relationships:
            triplet = GraphRelation(relationship.source.id, relationship.target.id, relationship.type)
            triples.append(triplet)
        return triples