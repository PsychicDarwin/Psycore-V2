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
        
        if context is None:
            context = []
        context_text = [item for item in context if isinstance(item, str)]
        context_attachments = [item for item in context if isinstance(item, Attachment) and item.attachment_type == AttachmentTypes.IMAGE]
        context_label = "context"
        context_image_count = len(context_attachments)
        context_role = "human"
        context_text_count = len(context_text)
        total_added_messages = 0
        if context_image_count > 0:
            new_history.append((context_role, ContentFormatter.prep_images(context_image_count,context_label)))
            total_added_messages += context_image_count
        if context_text_count > 0:
            new_history.append((context_role, ContentFormatter.prep_texts(context_text_count, context_label)))
            total_added_messages += context_text_count
        prompt_text = [item for item in prompt if isinstance(item, str)]
        prompt_attachments = [item for item in prompt if isinstance(item, Attachment) and item.attachment_type == AttachmentTypes.IMAGE]
        prompt_label = "prompt"
        prompt_role = "user"
        prompt_image_count = len(prompt_attachments)
        prompt_text_count = len(prompt_text)
        if prompt_image_count > 0:
            new_history.append((prompt_role, ContentFormatter.prep_images(prompt_image_count,prompt_label)))
            total_added_messages += prompt_image_count
        if prompt_text_count > 0:
            new_history.append((prompt_role, ContentFormatter.prep_texts(prompt_text_count, prompt_label)))
            total_added_messages += prompt_text_count

        langchain_prompt = {
            **ContentFormatter.map_image_data(
                [attachment.attachment_data for attachment in prompt_attachments],
                prompt_label),
            **ContentFormatter.map_image_data(
                [attachment.attachment_data for attachment in context_attachments],
                context_label),
            **ContentFormatter.map_text_data(
                prompt_text,
                prompt_label),
            **ContentFormatter.map_text_data(
                context_text,
                context_label)
        }

        if self.history is not None:
            self.history.chat_history.extend(new_history)
            template = self.history.create_template()
        else:
            extended_chat = [("system", self.system_prompt)]
            extended_chat.extend(new_history)
            template = ChatPromptTemplate.from_messages(extended_chat)
        
        llm_output = ContentFormatter.chat_to_model(template, self.wrapper, langchain_prompt)
        if self.history is not None:
            # We append output to chat, but acknowledge that since we added multiple messages with potential formatting, they'll need to be rewritten without langchain styling so we can reuse history
            self.history.append_output_to_chat(template, langchain_prompt, llm_output, total_added_messages)
        return llm_output
    
    def process_prompt_text(self, prompt : list[Union[str, Attachment]], context : list[Union[str, Attachment]] = None):
        return self.process_prompt(prompt, context).content
    
    def process_text_no_context(self, prompt : list[Union[str, Attachment]]):
        return self.process_prompt(prompt, []).content
