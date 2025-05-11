from .evaluator import Evaluator
from evaluate import load
from rouge_score import rouge_scorer
from src.data.s3_quick_fetch import S3QuickFetch

class RougeEvaluator(Evaluator, S3QuickFetch):
    def __init__(self, s3_quick_fetch: S3QuickFetch):
        super().__init__(s3_quick_fetch)
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        pass

    def evaluate(self, source: str, result: str):
        """
        Evaluates the similarity between the question and answer using Rouge embeddings.

        :param source: The question to evaluate against.
        :param result: The answer to evaluate.
        :return: The similarity score between the question and answer.
        """
        results = self.scorer.score(source, result)
        return results

    def evaluate_rag_result(self, result: str, rag_result: dict):
        summary = self.pull_summary(rag_result)
        rouge_result = {}
        if summary != "":
            rouge_result = self.evaluate(summary, result)

        rag_result["rouge_evaluation"] = rouge_result
        return rag_result

    def overall_value(self, output: str, additional_params: dict):
        """
        Iterates over all outputs for F1, precision, and recall.

        Averages all F1s and returns the average.
        """
        source = additional_params["document_data"]
        results = self.evaluate(source, output)
        f1 = results["f1"]
        return sum(f1) / len(f1)
    
    