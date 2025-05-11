from .evaluator import Evaluator
from src.kg import GraphCreator, GraphRelation, dict_data_to_relations
from src.data.s3_quick_fetch import S3QuickFetch
import json
class GraphEvaluator(Evaluator):
    def __init__(self, graph_creator: GraphCreator, s3_quick_fetch: S3QuickFetch,beta: float = 1.0):
        super().__init__(s3_quick_fetch)
        self.graph_creator = graph_creator
        self.beta = beta

    def convert_output_to_graph(self, output: str):
        """
        Converts the output into a graph representation using the GraphCreator.

        :param output: The output to convert.
        :return: The graph representation of the output.
        """
        return self.graph_creator.create_graph_json(output)
    
    def compare_graph_precision(self, retrieved_graph: list[GraphRelation], llm_graph: list[GraphRelation]) -> float:
        """
        Precision = True Positives / LLM Predictions
        Measures how many of the LLM's relations were based on what it actually saw.
        """
        if not llm_graph:
            return 0.0

        retrieved_set = set(retrieved_graph)
        llm_set = set(llm_graph)

        true_positives = retrieved_set.intersection(llm_set)
        return len(true_positives) / len(llm_set)
    
    def compare_graph_recall(self, full_truth_graph: list[GraphRelation], retrieved_graph: list[GraphRelation]) -> float:
        """
        Recall = True Positives / All True Relations
        Measures how well the retrieved chunk covered the true content.
        """
        if not full_truth_graph:
            return 0.0

        truth_set = set(full_truth_graph)
        retrieved_set = set(retrieved_graph)

        true_positives = truth_set.intersection(retrieved_set)
        return len(true_positives) / len(truth_set)
    
    def f_beta_score(self, precision: float, recall: float, beta: float = 1.0) -> float:
        if (precision + recall) == 0:
            return 0
        return (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)
    
    def compare_graph_f_beta(self, source_graph: list[GraphRelation], result_graph: list[GraphRelation], beta: float = 1.0) -> float:
        """
        Compares the F-beta score of two graphs using the recall and precision scores.
        :param source_graph: The source graph to compare against.
        :param result_graph: The result graph to compare.
        :param beta: The beta value for the F-beta score.
        :return: The F-beta score between the two graphs.
        """
        precision = self.compare_graph_precision(source_graph, result_graph)
        recall = self.compare_graph_recall(source_graph, result_graph)
        if (precision + recall) == 0:
            return 0
        return self.f_beta_score(precision, recall, beta)


    def evaluate(self, source: str, result: str):
        pass
    
    def overall_value(self, output: str, additional_params: dict):
        pass
        
    
     
    

    def evaluate_rag_result(self, result: str, rag_data: dict):
        graph_s3_path = rag_data["graph_path"]
        # Get the graph file from s3 bucket as text
        graph_text = self.s3_quick_fetch.fetch_text(graph_s3_path)
        if rag_data["type"] == "image" or rag_data["type"] == "attachment_image":
            text = self.pull_summary(rag_data)
        elif rag_data["type"] == "text":
            text = rag_data["text"]
        else:
            graph_text = None
        graph_evaluation = {}
        if graph_text is None:
            graph_evaluation = {
                "recall": -1,
                "precision": -1,
                "f_beta": -1,
                "beta": self.beta
            }
        else:
            json_graph = json.loads(graph_text)
            graph_data = dict_data_to_relations(json_graph)
            text_graph = self.graph_creator.create_graph_relations(text)
            llm_graph = self.graph_creator.create_graph_relations(result)

            recall = self.compare_graph_recall(graph_data, text_graph)
            precision = self.compare_graph_precision(graph_data, llm_graph)
            f_beta = self.compare_graph_f_beta(graph_data, llm_graph, self.beta)

            graph_evaluation = {
                "recall": recall,
                "precision": precision,
                "f_beta": f_beta,
                "beta": self.beta
                }
        rag_data["graph_evaluation"] = graph_evaluation
        return rag_data
