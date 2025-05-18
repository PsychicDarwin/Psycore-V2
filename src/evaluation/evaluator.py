from abc import ABC, abstractmethod
from src.data.s3_quick_fetch import S3QuickFetch
from src.main import IterativeStage
class Evaluator(ABC):
    @abstractmethod
    def __init__(self,iterative_stage: IterativeStage):
        self.iterative_stage = iterative_stage
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
