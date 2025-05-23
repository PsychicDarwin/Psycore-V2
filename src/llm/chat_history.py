from src.llm.content_formatter import ContentFormatter
from langchain.prompts import ChatPromptTemplate

class ChatHistory:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.chat_history = [
            ("system", system_prompt)
        ]
    
    def add_image_prep(self, count: int, role: str = "user", label: str = None):
        self.chat_history.append(
            (role, ContentFormatter.prep_images(count, label))
        )

    def add_text(self, text: str, role: str = "user"):
        self.chat_history.append(
            (role, text)
        )

    def create_template(self):
        return ChatPromptTemplate.from_messages(self.chat_history)

    def append_output_to_chat(self, template: ChatPromptTemplate, prompt_dict: dict, llm_output: dict, total_added_messages: int = 1):
        chat_output = template.invoke(prompt_dict)
        self.chat_history = chat_output.messages
        self.chat_history.append(("ai", llm_output.content))
        
