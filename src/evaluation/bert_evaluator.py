from .evaluator import Evaluator
from evaluate import load

class BERTEvaluator(Evaluator):
    def __init__(self, iterative_stage):
        super().__init__(iterative_stage)
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
    

    def evaluate_rag_result(self, result: str, rag_result: dict):
        summary = self.iterative_stage.chunk_summaries[rag_result["vector_id"]]["summary"]
        bertscore_result = {}
        if summary != "":
            bertscore_result = self.evaluate(summary, result)
        rag_result["bertscore_evaluation"] = bertscore_result
        return rag_result