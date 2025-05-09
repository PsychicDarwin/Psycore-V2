from .evaluator import Evaluator
from evaluate import load

class BERTEvaluator(Evaluator):
    def __init__(self):
        self.bertscore = load("bertscore", module_type="metric")
        pass

    def evaluate(self, source: str, result: str):
        """
        Evaluates the similarity between the question and answer using BERT embeddings.

        :param source: The question to evaluate against.
        :param result: The answer to evaluate.
        :return: The similarity score between the question and answer.
        """
        results = self.bertscore.compute(predictions=[result], references=[source], lang="en")
        return results
    
    def overall_value(self, output: str, additional_params: dict):
        """
        Iterates over all outputs for F1, precision, and recall.

        Averages all F1s and returns the average.
        """
        source = additional_params["document_data"]
        results = self.evaluate(source, output)
        f1 = results["f1"]
        return sum(f1) / len(f1)