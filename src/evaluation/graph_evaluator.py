from .evaluator import Evaluator
from src.kg import GraphCreator, GraphRelation

class GraphEvaluator(Evaluator):
    def __init__(self, graph_creator: GraphCreator):
        self.graph_creator = graph_creator
        pass

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
        return (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)


    def evaluate(self, source: str, result: str):
        pass
    
    def overall_value(self, output: str, additional_params: dict):
        pass
    