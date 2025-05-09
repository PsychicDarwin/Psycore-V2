from .chat_history import ChatHistory
from .wrappers import ChatModelWrapper 
from .content_formatter import ContentFormatter
from src.data.attachments import Attachment, AttachmentTypes
from typing import Union
from langchain.prompts import ChatPromptTemplate

class ChatAgent:
    def __init__(self, model: ChatModelWrapper, system_prompt: str, history: bool = False):
        self.wrapper = model
        self.system_prompt = system_prompt
        if history:
            self.history = ChatHistory(system_prompt)
        else:
            self.history = None

    def process_prompt(self, prompt : list[Union[str, Attachment]], context : list[Union[str, Attachment]] = None):
        new_history = []
        # Split context into text inputs and attachments
        context_text = [item for item in context if isinstance(item, str)]
        context_attachments = [item for item in context if isinstance(item, Attachment) and item.attachment_type == AttachmentTypes.IMAGE]
        context_label = "context_image"
        context_image_count = len(context_attachments)
        context_role = "context"
        total_added_messages = 0
        if context_image_count > 0:
            new_history.append(self.history.add_image_prep(context_image_count,context_role,context_label))
            total_added_messages += context_image_count
        for item in context_text:
            new_history.append(self.history.add_text(item,context_role))
            total_added_messages += 1

        prompt_text = [item for item in prompt if isinstance(item, str)]
        prompt_attachments = [item for item in prompt if isinstance(item, Attachment) and item.attachment_type == AttachmentTypes.IMAGE]
        prompt_label = "prompt_image"
        prompt_role = "user"
        prompt_image_count = len(prompt_attachments)
        if prompt_image_count > 0:
            new_history.append(self.history.add_image_prep(prompt_image_count,prompt_role,prompt_label))
            total_added_messages += prompt_image_count
        for item in prompt_text:
            new_history.append(self.history.add_text(item,prompt_role))
            total_added_messages += 1

        langchain_prompt = {
            **ContentFormatter.map_image_data(
                [attachment.attachment_data for attachment in prompt_attachments],
                                            prompt_label),
            **ContentFormatter.map_image_data(
                [attachment.attachment_data for attachment in context_attachments],
                                            context_label)}

        if self.history is not None:
            self.history.chat_history.extend(new_history)
            template = self.history.create_template()
        else:
            template = ChatPromptTemplate.from_messages([("system", self.system_prompt)].extend(new_history))
        
        llm_output = self.wrapper.invoke(template, langchain_prompt)
        if self.history is not None:
            # We append output to chat, but acknowledge that since we added multiple messages with potential formatting, they'll need to be rewritten without langchain styling so we can reuse history
            self.history.append_output_to_chat(template, langchain_prompt, llm_output, total_added_messages)
        return llm_output
    
