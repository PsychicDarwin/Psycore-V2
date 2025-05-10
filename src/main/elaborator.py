from src.llm.chat_agent import ChatAgent
from abc import ABC, abstractmethod
from src.llm.wrappers import ChatModelWrapper
class Elaborator(ABC):
    @abstractmethod
    def __init__(self, wrapper: ChatModelWrapper, system_prompt: str = None):
        if system_prompt is None:
            system_prompt = """
            You are a friendly and helpful assistant."""
        self.system_prompt = system_prompt
        self.agent = ChatAgent(wrapper, system_prompt, history=False)
        pass

    def elaborate(self, prompt: str) -> str:
        return self.agent.process_text_no_context([prompt])


class RAGElaborator(Elaborator):
    def __init__(self, wrapper: ChatModelWrapper):
        super().__init__(wrapper, """You are an expert RAG/LLM prompt engineer. Your task is to take all user prompts and enhance them to be more effective before it is embedded for RAG. This is to combat issues of missing data like 2020 - 2024 containing 2020 2021,2022,2023,2024 and similar issues.
        
        Extract all keywords and make every detail explicit to enhance the retrieval chances of the RAG.
        Return the elaborated prompt text, nothing else.
        """)
    
class UserPromptElaboration(Elaborator):
    def __init__(self, wrapper: ChatModelWrapper):
        super().__init__(wrapper, """You are an expert prompt engineer. Your task is to elaborate and improve the following prompt 
            to make it more effective for a language model. Add specific instructions, context, and formatting 
            that will help the model provide a better response.

            Provide an elaborated version of this prompt that will get better results from an LLM. 
            Given a users prompt, return the improved prompt text, nothing else.
        """)