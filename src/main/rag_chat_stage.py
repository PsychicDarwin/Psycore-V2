from src.llm.wrappers import ChatModelWrapper
from src.llm.chat_agent import ChatAgent
from src.data.s3_handler import S3Handler
from src.data.s3_quick_fetch import S3QuickFetch

class RAGChatStage:
    def __init__(self, wrapper: ChatModelWrapper, s3_handler: S3Handler):
        self.chat_agent = ChatAgent(wrapper,history=True, system_prompt="""
You are an information retrieval and vertification assistant, you will recieve a variety of source documents and will be asked queries by a user. Your job is to answer user queries as accurately as possible given the information you have available and to prevent halluciations where you can by double checking your sources.""")
        self.s3_handler = s3_handler
        self.s3_quick_fetch = S3QuickFetch(s3_handler)

    def chat(self, prompt: str, rag_results: list) -> str:
        context = []
        for result in rag_results:
            if result["type"] == "text":
                context.append(result["text"])
            elif result["type"] == "image" or result["type"] == "attachment_image":
                context.append(self.s3_quick_fetch.get_image(result["image_path"]))
            else:
                raise ValueError(f"Unknown result type: {result['type']}")
        prompt = [prompt]
        return self.chat_agent.process_prompt(prompt, context)
    

