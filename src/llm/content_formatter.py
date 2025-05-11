from langchain_core.prompts import ChatPromptTemplate
from src.llm.wrappers import ChatModelWrapper
from langchain_text_splitters import TokenTextSplitter
IMAGE_LABEL = "image"
TEXT_LABEL = "text"
class ContentFormatter:

    def format_base_chat(system_prompt) -> list:
        return [("system", system_prompt)]

    def add_format_to_chat(chat, formatted_prompt) -> list:
        """
        Adds formatted prompt to chat history.
        Args:
            chat (list): Chat history.
            formatted_prompt (str): Formatted prompt.
        Returns:
            list: Updated chat history.
        """
        chat.append(("user", formatted_prompt))
        return chat

    def prep_texts(count: int, label: str = "") -> dict:
        """
        Formats text for LLM input.
        Args:
            text (str): Input text.
        Returns:
            dict: Formatted text data.
        """
        label = label + TEXT_LABEL
        return_val = []
        for i in range(count):
            return_val.append({
                "type": "text",
                "text": "{" + f"{label}{i}" + "}"
            })
        return return_val
    
    def prep_images(count: int,label: str = "") -> list[dict]:
        """
        Formats a base64 image string for LLM input.
        Args:
            base64_image (str): Base64 encoded image string.
        Returns:
            dict: Formatted image data.
        """
        label = label + IMAGE_LABEL
        return_val = []
        for i in range(count):
            return_val.append({
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64,{" + f"{label}{i}" + "}"
                }
            })
        return return_val

    def map_image_data(image_data: list, label: str = "") -> dict:
        """
        Maps image data to a format suitable for LLM input.
        Args:
            image_data (list): List of image data.
        Returns:
            dict: Mapped image data.
        """
        mappings = {}
        label = label + IMAGE_LABEL
        for i, base64_image in enumerate(image_data):
            mappings[f"{label}{i}"] = base64_image
        return mappings
    
    def map_text_data(text_data: list, label: str = "") -> dict:
        mappings = {}
        label = label + TEXT_LABEL
        for i, text in enumerate(text_data):
            mappings[f"{label}{i}"] = text
        return mappings
    
    def chat_to_template(chat):
        return ChatPromptTemplate.from_messages(chat)
    
    def format_prompt(prompt_text = None, image_data = None) -> dict:
        """
        Formats the prompt for LLM input.
        Args:
            prompt_text (str): Input text.
            image_data (list): List of image data.
        Returns:
            dict: Formatted prompt data.
        """
        prompt_map = {}
        if image_data and len(image_data) > 0:
            prompt_map = ContentFormatter.map_image_data(image_data)
        if prompt_text:
            prompt_map["prompt"] = prompt_text
        return prompt_map


    def chat_to_model(template: ChatPromptTemplate, wrapper: ChatModelWrapper, prompt_data: dict) -> dict:
        chain = template | wrapper.model
        response = chain.invoke(prompt_data)
        return response
        

    def append_to_chat(template: ChatPromptTemplate, chat_array: dict, prompt_data: dict, llm_output: dict) -> dict:
        """
        Appends the model output to the chat history.
        Args:
            template (ChatPromptTemplate): The chat template used to convert last prompt into a string.
            chat_array (dict): The chat history to be updated.
            prompt_data (dict): The prompt data to be used for the conversion of last prompt into a string.
            llm_output (dict): The model output to be appended to the chat history as AI response.
        """
        chat_output = template.invoke(prompt_data)
        # After conversion we add the previous chat data with formatting to history
        chat_array[-1] = ("user", chat_output.messages.pop().content)
        # We add Model output to the chat history
        chat_array.append(("ai", llm_output.content))
        return chat_array
        
        
    def chunk_text(text, chunk_size: int = None, chunk_overlap: int = None) -> list:
        """Chunk the text into smaller pieces for embedding."""
        text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(text)
        return chunks