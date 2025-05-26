import base64
from .data_helper import split_audio_file, clean_temp_files
import whisper
import logging
from enum import Enum
from PIL import Image
from io import BytesIO
from src.llm.wrappers import ChatModelWrapper
from src.llm.content_formatter import ContentFormatter
from .filereader import FileReader
from src.system_manager import LoggerController

# Initialize the logger
logger = LoggerController.get_logger()

MAX_LLM_IMAGE_PIXELS = 512

class AttachmentTypes(Enum):
    IMAGE = 1
    AUDIO = 2
    VIDEO = 3
    FILE =4 

    @staticmethod
    def from_filename(filename: str) -> "AttachmentTypes":
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif")):
            return AttachmentTypes.IMAGE
        elif filename.lower().endswith((".mp3", ".wav", ".flac")):
            return AttachmentTypes.AUDIO
        elif filename.lower().endswith((".mp4", ".avi", ".mov")):
            return AttachmentTypes.VIDEO
        else:
            return AttachmentTypes.FILE
        
class Attachment:
    def __init__(self, attachment_type: AttachmentTypes, attachment_data: str, needs_extraction: bool = False, additional_data: dict = None):
        self.attachment_type = attachment_type
        self.attachment_data = attachment_data
        self.needs_extraction = needs_extraction
        self.additional_data = additional_data if additional_data else {}


    def image_to_attachment(image_information: dict, additional_data: dict = None):
        """
        Converts an extracted image dict (from a PDF) to an Attachment object.
        :param image_information: Dictionary containing image information.
        :param additional_data: Additional data to be stored with the image.
        :return: Attachment object.
        """
        if image_information is None:
            raise ValueError("Image information cannot be None")
        attachment_data = image_information.get("image")
        if attachment_data is None:
            raise ValueError("Attachment data cannot be None")
        attachment_type = AttachmentTypes.IMAGE
        image_information.pop("image")
        additional_data = dict(list(additional_data.items()) + list(image_information.items()))
        return Attachment(attachment_type, attachment_data, needs_extraction=False, additional_data=additional_data)
    
    def extract(self):
        if self.needs_extraction:
            try:
                self.needs_extraction = False
                if self.attachment_type == AttachmentTypes.AUDIO:
                    self._extract_audio()
                elif self.attachment_type == AttachmentTypes.VIDEO:
                    self._extract_video()
                elif self.attachment_type == AttachmentTypes.IMAGE:
                    self._extract_image()
                elif self.attachment_type == AttachmentTypes.FILE:
                    self._extract_contents()
                else:
                    self.needs_extraction = True
            except Exception as e:
                logger.error(f"Error extracting attachment: {str(e)}")
                self.needs_extraction = True

    def _extract_image(self):
        try: 
            with Image.open(self.attachment_data) as img:
                img = img.convert("RGB")
                if (MAX_LLM_IMAGE_PIXELS != None):
                    img = img.resize((MAX_LLM_IMAGE_PIXELS, MAX_LLM_IMAGE_PIXELS))
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                self.attachment_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
                self.additional_data.update({
                    "width": img.width,
                    "height": img.height,
                    "format": img.format
                })
        except Exception as e:
            logger.error(f"Failed to process image: {str(e)}")
            raise FailedExtraction(self, f"Failed to process image: {str(e)}")

    def text_summary(self, wrapper: ChatModelWrapper):
        if self.attachment_type != AttachmentTypes.IMAGE:
            # If the attachment is not an image, we don't need to process it as if regular text, or audio, it's already in text format
            return self.attachment_data
        else:
            chat = ContentFormatter.format_base_chat("You are a description generator. You will describe in as much detail as possible the image you are given. You will only respond with the description of the image.")
            chat = ContentFormatter.add_format_to_chat(chat, ContentFormatter.prep_images(1,))
            prompt_data = ContentFormatter.format_prompt(image_data=[self.attachment_data])
            template = ContentFormatter.chat_to_template(chat)
            description = ContentFormatter.chat_to_model(template, wrapper, prompt_data)
            return description.content


    def _extract_audio(self):
        # We process audio files by splitting them into smaller chunks and transcribing them using Whisper
        # Not all MLLM models support audio input, so we just convert it to text
        try: 
            chunk_size_mb = 25
            chunks = split_audio_file(self.attachment_data, chunk_size_mb)
            model = whisper.load_model("medium")
            transcriptions = []
            for chunk in chunks:
                result = model.transcribe(chunk, fp16=False)
                transcriptions.append(result["text"])
            
            self.attachment_data = " ".join(transcriptions)
            
            clean_temp_files(chunks)
        except Exception as e:
            logger.error(f"Failed to process audio: {str(e)}")
            raise FailedExtraction(self, f"Failed to process audio: {str(e)}")
            
    def _extract_contents(self):
        file_extension = self.attachment_data.split('.')[-1].lower()
        TXT_TYPES = ['txt','md','json','xml','yaml','csv']
        DOC_TYPES = ['doc','docx']
        PDF_TYPES = ['pdf']
        XLS_TYPES = ['xls','xlsx']
        ALL_TYPES = TXT_TYPES + DOC_TYPES + PDF_TYPES + XLS_TYPES
        if file_extension in ALL_TYPES:
            # Use the appropriate file reader based on the file extension
            if file_extension in TXT_TYPES:
                self.attachment_data = FileReader.extract_txt(self.attachment_data)
            elif file_extension in DOC_TYPES:
                self.attachment_data = FileReader.extract_docx(self.attachment_data)
            elif file_extension in XLS_TYPES:
                self.attachment_data = FileReader.extract_xlsx(self.attachment_data)
            elif file_extension in PDF_TYPES:
                self.attachment_data = FileReader.extract_pdf(self.attachment_data)
            else:
                self.attachment_data = None

            if self.attachment_data is not None:
                return self.attachment_data
        else:
            logger.error(f"Unsupported file type: {file_extension}")
            raise FailedExtraction(self, f"Unsupported file type: {file_extension}")

    def _extract_video(self):
        raise NotImplementedError("Video extraction is not implemented yet.")

    





class FailedExtraction(Exception):
    def __init__(self, attachment: Attachment, message: str):
        super().__init__(f"""
Failed to extract attachment
Attachment type: {attachment.attachment_type.name}
Error: {message}
        """)
        self.attachment = attachment
        