from src.vector_database.vector_service import VectorService
from src.data.s3_handler import S3Handler, S3Bucket
from src.data.attachments import Attachment, AttachmentTypes
from src.vector_database import Embedder
from src.llm.wrappers import ChatModelWrapper
from src.kg.graph_creator import GraphCreator
class FilePreprocessor:

    def __init__(self, s3_handler: S3Handler, vector_database: VectorService, embedder : Embedder, imageConverter: ChatModelWrapper, graph_creator: GraphCreator):
        self.s3_handler = s3_handler
        self.vector_database = vector_database
        self.embedder = embedder
        self.imageConverter = imageConverter

    def process_file(self, file_path: str, additional_data : dict):
        """
        Process the file and add it to the vector database.
        :param file_path: Path to the file.
        :param additional_data: Additional data to be stored with the file.
        """
        attachment_format = Attachment.get_attachment_type(file_path)
        attachment_file = Attachment(file_path, attachment_format, needs_extraction=True, additional_data=None)
        attachment_file.extract()
        if attachment_file.needs_extraction:
            print("Failed to read attachment")
        else:
            if type(attachment_file.attachment_data) == str:
                # If the attachment is a string, it means it's a text file or bbase64 image
                data = attachment_file.attachment_data
            elif type(attachment_file.attachment_data) == dict:
                data = attachment_file.attachment_data["text"]
                additional_data = ""
                if "images" in attachment_file.attachment_data.keys():
                    attachment_images = [
                        Attachment.image_to_attachment(image, additional_data=additional_data) for image in attachment_file.attachment_data["images"]
                    ]
                    image_text = [
                        file._text_summary(self.imageConverter) for file in attachment_images
                    ]
                    for image in attachment_images:
                        image.attachment_data["summary"] = image_text
                    

            else:
                raise ValueError("Unknown attachment data type")
    def process_additional_image(self, image: Attachment):

        



    def process_files(self, files):
        for file in files:
            # We download the file and go through our temp_download s3 process
            self.s3_handler.download_to_temp_and_process(
                bucket=S3Bucket.DOCUMENTS,
                file_key=file,
                process_func=self.process_file
            )