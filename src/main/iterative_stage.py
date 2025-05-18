from src.kg.graph_creator import GraphCreator, GraphRelation, dict_data_to_relations, remove_dup_relations
from src.data.s3_quick_fetch import S3QuickFetch
from src.system_manager.LoggerController import LoggerController
import json
class IterativeStage:
    def __init__(self, quick_fetch: S3QuickFetch, graphModel: GraphCreator, threshold: float = 0.5, rag_results = []):
        self.s3_quick_fetch = quick_fetch
        self.graphModel = graphModel
        self.threshold = threshold
        self.logger = LoggerController.get_logger()
        
        self.doc_graphs = {}
        for i in range(len(rag_results)):
            # We get the document_path and the graph_path for each and make pairings
            if rag_results[i]["document_path"] not in self.doc_graphs:
                # Get the graph file from s3 bucket as text
                self.doc_graphs[rag_results[i]["document_path"]] = self.s3_quick_fetch.fetch_text(rag_results[i]["graph_path"])
                self.doc_graphs[rag_results[i]["document_path"]] = json.loads(self.doc_graphs[rag_results[i]["document_path"]])
                self.doc_graphs[rag_results[i]["document_path"]] = dict_data_to_relations(self.doc_graphs[rag_results[i]["document_path"]])
            
        self.mega_doc_graph = []
        for key, value in self.doc_graphs.items():
            self.logger.info(f"Processing graph document {key}")
            for i in range(len(value)):
                if value[i] not in self.mega_doc_graph:
                    self.mega_doc_graph.extend(value)
        self.mega_doc_graph = remove_dup_relations(self.mega_doc_graph)
        self.chunk_summaries = {}
        for i in range(len(rag_results)):
            self.chunk_summaries[rag_results[i]["vector_id"]] = {
                "summary": self.s3_quick_fetch.pull_summary(rag_results[i])
            }
            self.chunk_summaries[rag_results[i]["vector_id"]]["graph"] = self.graphModel.create_graph_relations(self.chunk_summaries[rag_results[i]["vector_id"]]["summary"])
        self.mega_chunk_graph = []
        for key, value in self.chunk_summaries.items():
            self.logger.info(f"Processing graph document {key}")
            self.mega_chunk_graph.extend(value["graph"])
        self.mega_chunk_graph = remove_dup_relations(self.mega_chunk_graph)
    
    def relation_percentage(self, item_relations: list[GraphRelation], all_relations: list[GraphRelation]) -> float:
        # Go through item relation, check if it exists in all_relations (treating all_relations as list of valid relations)
        # We can then return the percentage of relations that are valid from item_relations
        valid_items = 0 
        for i in range(len(item_relations)):
            if item_relations[i] in all_relations:
                valid_items += 1
        return valid_items / len(item_relations) if len(item_relations) > 0 else 0.0
    
    def return_missing_relations(self, item_relations: list[GraphRelation], all_relations: list[GraphRelation]) -> list[GraphRelation]:
        # Go through item relation, check if it exists in all_relations (treating all_relations as list of valid relations)
        # We can then return the percentage of relations that are valid from item_relations
        missing_items = []
        for i in range(len(item_relations)):
            if item_relations[i] not in all_relations:
                missing_items.append(item_relations[i])
        return missing_items
            
    
    def decision_maker(self, rag_results: list[dict], rag_chat_results):
        """
        This function is used to make a decision based on the rag_result and additional_params.
        It can be overridden by subclasses to provide specific decision-making logic.

        :param rag_result: The result of the RAG process.
        :param additional_params: Additional parameters for decision-making.
        :return: The decision made based on the rag_result and additional_params.
        """
        llm_graph = self.graphModel.create_graph_relations(rag_chat_results.content)
            
        valid_relations = self.relation_percentage(llm_graph, self.mega_doc_graph)
        if valid_relations < self.threshold:
            missing_relations = self.return_missing_relations(llm_graph, self.mega_chunk_graph)
            self.logger.info(f"Valid relations: {valid_relations} is below threshold: {self.threshold}")
            return (valid_relations, missing_relations, llm_graph)
        else:
            self.logger.info(f"Valid relations: {valid_relations} is above threshold: {self.threshold}")
            return (valid_relations, [], llm_graph)