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
    
    def compare_graph_recall(self, source_graph: list[GraphRelation], result_graph: list[GraphRelation]) -> float:
        """
        Compares the recall of two graphs using amount of entries in the result graph as a percentage of the source graph.

        :param source_graph: The source graph to compare against.
        :param result_graph: The result graph to compare.
        :return: The recall score between the two graphs.
        """
        source_graph_count = len(source_graph)
        result_graph_count = len(result_graph)
        if (result_graph_count > source_graph_count):
            result_count = source_graph_count
        else:
            result_count = result_graph_count
        return result_count / source_graph_count if source_graph_count > 0 else 0
    
    def compare_graph_precision(self, source_graph: list[GraphRelation], result_graph: list[GraphRelation]) -> float:
        """
        Compares the precision of two graphs using amount of entries in the source graph as a percentage of the result graph, checking overlap between each relation.
        :param source_graph: The source graph to compare against.
        :param result_graph: The result graph to compare.
        :return: The precision score between the two graphs.
        """
        matched_count = 0
        source_graph_count = len(source_graph)
        for source_relation in source_graph:
            for result_relation in result_graph:
                if source_relation == result_relation:
                    matched_count += 1
                    break
        return matched_count / source_graph_count if source_graph_count > 0 else 0
    
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
        if graph_text is None:
            return (-1, -1, -1)
        else:
            json_graph = json.loads(graph_text)
            graph_data = dict_data_to_relations(json_graph)
            llm_graph = self.graph_creator.create_graph_json(result)

            recall = self.compare_graph_recall(graph_data, llm_graph)
            precision = self.compare_graph_precision(graph_data, llm_graph)
            f_beta = self.compare_graph_f_beta(graph_data, llm_graph, self.beta)

            graph_evaluation = {
                "recall": recall,
                "precision": precision,
                "f_beta": f_beta,
                "beta": self.beta}
            rag_data["graph_evaluation"] = graph_evaluation
            return rag_data
