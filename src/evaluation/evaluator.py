from abc import ABC, abstractmethod
from src.data.s3_quick_fetch import S3QuickFetch
class Evaluator(ABC):
    @abstractmethod
    def __init__(self, s3_quick_fetch: S3QuickFetch):
        self.s3_quick_fetch = s3_quick_fetch
        pass

    @abstractmethod
    def evaluate_rag_result(self, result: str, rag_data: dict):
        return rag_data



    @abstractmethod
    def overall_value(self, output: str, additional_params : dict):
        """
        Evaluates the overall value of the output based on additional parameters.
        This method can be overridden by subclasses to provide specific evaluation logic.

        :param output: The output to evaluate.
        :param additional_params: Additional parameters for evaluation.
        Example of additional_params:
        {
            "document_data" : ["text1", "text2"]
        }
        
        :return: The overall value of the output.
        """
        return 0
    

    def pull_summary(self, rag_data: dict):
        if rag_data["type"] == "text":
            return rag_data["text"]
        elif rag_data["type"] == "image" or rag_data["type"] == "attachment_image":
            if rag_data["summary_path"] is not None:
                summary = self.s3_quick_fetch.fetch_text(rag_data["summary_path"])
                return summary
        return ""
