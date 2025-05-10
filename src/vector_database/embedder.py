from abc import ABC, abstractmethod
from typing import BinaryIO, Union
from PIL import Image
from langchain_text_splitters import TokenTextSplitter
from numpy import ndarray
import base64
from io import BytesIO

class Embedder(ABC):
    @abstractmethod
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 24, dimension_output: int = 512):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.dimension_output = dimension_output

    def chunk_text(self,text) -> list:
        """Chunk the text into smaller pieces for embedding."""
        text_splitter = TokenTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = text_splitter.split_text(text)
        return chunks

    @abstractmethod
    def text_to_embedding(chunk_data : str) -> ndarray:
        """Convert chunk data to embedding."""
        pass

    def image_to_base64(self, image: Union[Image.Image, BinaryIO]) -> str:
        if isinstance(image, Image.Image):
            # Convert to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            # Save to BytesIO in JPEG format
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            return base64.b64encode(img_byte_arr).decode("utf-8")
        elif isinstance(image, BytesIO):
            # If it's already a BytesIO, just read and encode
            image.seek(0)
            return base64.b64encode(image.read()).decode("utf-8")
        else:
            raise ValueError("Unsupported image type. Must be PIL.Image or BytesIO")

    @abstractmethod
    def image_to_embedding(image: Union[Image.Image, BinaryIO]) -> ndarray:
        """Convert image data to embedding."""
        pass