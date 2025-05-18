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
                try:
                    # Get the graph file from s3 bucket as text
                    graph_text = self.s3_quick_fetch.fetch_text(rag_results[i]["graph_path"])
                    # If the fetch failed and returned empty string, skip this document
                    if not graph_text:
                        self.logger.warning(f"Could not fetch graph for {rag_results[i]['document_path']}. Skipping.")
                        # Add an empty list as a placeholder for this document path
                        self.doc_graphs[rag_results[i]["document_path"]] = []
                        continue
                    
                    # Try to parse the JSON and convert to relations
                    try:
                        graph_json = json.loads(graph_text)
                        self.doc_graphs[rag_results[i]["document_path"]] = dict_data_to_relations(graph_json)
                    except (json.JSONDecodeError, TypeError) as e:
                        self.logger.warning(f"Could not parse graph JSON for {rag_results[i]['document_path']}. Error: {str(e)}")
                        # Add an empty list as a placeholder for this document path
                        self.doc_graphs[rag_results[i]["document_path"]] = []
                except Exception as e:
                    self.logger.warning(f"Unexpected error processing graph for {rag_results[i]['document_path']}. Error: {str(e)}")
                    # Add an empty list as a placeholder for this document path
                    self.doc_graphs[rag_results[i]["document_path"]] = []
            
        self.mega_doc_graph = []
        for key, value in self.doc_graphs.items():
            self.logger.info(f"Processing graph document {key}")
            if value:  # Only process if value is not empty
                for i in range(len(value)):
                    if value[i] not in self.mega_doc_graph:
                        self.mega_doc_graph.extend(value)
        self.mega_doc_graph = remove_dup_relations(self.mega_doc_graph)
        
        self.chunk_summaries = {}
        for i in range(len(rag_results)):
            try:
                summary = self.s3_quick_fetch.pull_summary(rag_results[i])
                if not summary:
                    self.logger.warning(f"No summary found for vector_id {rag_results[i]['vector_id']}. Using placeholder.")
                    summary = "[No summary available]"
                
                self.chunk_summaries[rag_results[i]["vector_id"]] = {
                    "summary": summary
                }
                # Create graph relations from the summary
                try:
                    self.chunk_summaries[rag_results[i]["vector_id"]]["graph"] = self.graphModel.create_graph_relations(summary)
                except Exception as e:
                    self.logger.warning(f"Error creating graph relations for {rag_results[i]['vector_id']}. Error: {str(e)}")
                    self.chunk_summaries[rag_results[i]["vector_id"]]["graph"] = []
            except Exception as e:
                self.logger.warning(f"Error processing summary for {rag_results[i]['vector_id']}. Error: {str(e)}")
                self.chunk_summaries[rag_results[i]["vector_id"]] = {"summary": "[Error]", "graph": []}
        
        self.mega_chunk_graph = []
        for key, value in self.chunk_summaries.items():
            self.logger.info(f"Processing graph document {key}")
            if "graph" in value and value["graph"]:  # Check if graph exists and is not empty
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
        try:
            llm_graph = self.graphModel.create_graph_relations(rag_chat_results.content)
            
            # Handle case where mega_doc_graph might be empty
            if not self.mega_doc_graph:
                self.logger.warning("No document graph relations found. Cannot validate LLM output.")
                return (1.0, [], llm_graph)  # Assume valid (1.0) if we have no criteria to validate against
                
            valid_relations = self.relation_percentage(llm_graph, self.mega_doc_graph)
            if valid_relations < self.threshold:
                # Handle case where mega_chunk_graph might be empty
                if not self.mega_chunk_graph:
                    self.logger.warning("No chunk graph relations found. Cannot identify missing relations.")
                    missing_relations = []
                else:
                    missing_relations = self.return_missing_relations(llm_graph, self.mega_chunk_graph)
                    
                self.logger.info(f"Valid relations: {valid_relations} is below threshold: {self.threshold}")
                return (valid_relations, missing_relations, llm_graph)
            else:
                self.logger.info(f"Valid relations: {valid_relations} is above threshold: {self.threshold}")
                return (valid_relations, [], llm_graph)
        except Exception as e:
            self.logger.error(f"Error in decision_maker: {str(e)}")
            # Return a default response in case of error
            return (1.0, [], [])