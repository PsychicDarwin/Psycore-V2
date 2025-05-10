from src.vector_database.vector_service import VectorService
from src.data.s3_handler import S3Handler, S3Bucket
from src.data.attachments import Attachment, AttachmentTypes
from src.vector_database import Embedder
from src.llm.wrappers import ChatModelWrapper
from src.kg.graph_creator import GraphCreator
import base64, json
from io import BytesIO
from PIL import Image
from src.system_manager import LoggerController

logger = LoggerController.get_logger()

class FilePreprocessor:

    def __init__(self, s3_handler: S3Handler, vector_database: VectorService, embedder : Embedder, imageConverter: ChatModelWrapper, graph_creator: GraphCreator):
        logger.debug("Entering FilePreprocessor.__init__")
        self.s3_handler = s3_handler
        self.vector_database = vector_database
        self.embedder = embedder
        self.imageConverter = imageConverter
        self.graphModel = graph_creator
        logger.debug("Exiting FilePreprocessor.__init__")

    def process_file(self, file_path: str, additional_data : dict):
        logger.debug("Entering process_file with file_path=%s, additional_data=%s", file_path, additional_data)
        """
        Process the file and add it to the vector database.
        :param file_path: Path to the file.
        :param additional_data: Additional data to be stored with the file.
        """
        attachment_file = Attachment(AttachmentTypes.from_filename(file_path),file_path, needs_extraction=True, additional_data=None)
        attachment_file.extract()
        bucket_name, document_name, graph_path = None, None, None
        if additional_data is not None and "key" in additional_data.keys():
            # Split the document name and bucket name
            bucket_name = S3Bucket.DOCUMENTS.value
            document_name = additional_data["key"]
            graph_path = f"{document_name}.json"
        if attachment_file.needs_extraction:
            print("Failed to read attachment")
        else:
            appended_data = ""
            data = ""
            if type(attachment_file.attachment_data) is str:
                # If the attachment is a string, it means it's a text file or base64 image
                data = attachment_file.attachment_data

                if attachment_file.attachment_type == AttachmentTypes.IMAGE:
                    # Base64 to BytesIO binary
                    binary_image = BytesIO(base64.b64decode(data))
                    try:
                        # Convert BytesIO to PIL Image for embedding
                        embedding_ready_image = Image.open(binary_image)
                        embedded_image = self.embedder.image_to_embedding(embedding_ready_image)
                        embedding_ready_image.close()
                        
                        # Reset BytesIO position for S3 upload
                        binary_image.seek(0)
                        image_s3_uri = self.s3_handler.upload_image(document_name, binary_image)
                        
                        self.vector_database.add_data(embedded_image, {
                            "document_path": f"s3://{S3Bucket.DOCUMENTS.value}/{additional_data['key']}",
                            "graph_path": f"s3://{S3Bucket.GRAPHS.value}/{graph_path}",
                            "image_path": image_s3_uri,
                            "type": "image",
                        })
                        summary = self.imageConverter._text_summary(binary_image)
                        self.s3_handler.upload_document_summary(document_name, summary)
                        graph = self.graphModel.create_graph_dict(summary)
                        self.s3_handler.upload_graph(document_name, json.dumps(graph))
                    finally:
                        binary_image.close()
                else:
                    # If the attachment is a text file, we chunk it and add it to the vector database
                    chunked_data = self.embedder.chunk_text(data)
                    for chunk in chunked_data:
                        self.vector_database.add_data(chunk, {
                            "document_path": f"s3://{S3Bucket.DOCUMENTS.value}/{additional_data['key']}",
                            "graph_path": f"s3://{S3Bucket.GRAPHS.value}/{graph_path}",
                            "text": data,
                            "type": "text"
                        })
                    self.s3_handler.upload_document_text(document_name, data, file_type="summary")
                    graph = self.graphModel.create_graph_dict(data)
                    self.s3_handler.upload_graph(document_name, json.dumps(graph))

            elif type(attachment_file.attachment_data) is dict:
                data = attachment_file.attachment_data["text"]
                if type(data) is list:
                    data = "\n".join(data)
                data = data.replace("\n", " ")
                if "images" in attachment_file.attachment_data.keys():
                    attachment_images = [
                        Attachment.image_to_attachment(image, additional_data=additional_data) for image in attachment_file.attachment_data["images"]
                    ]
                    image_text = [
                        file._text_summary(self.imageConverter) for file in attachment_images
                    ]
                    for i, image in enumerate(attachment_images):
                        image.additional_data["summary"] = image_text[i]
                        image.additional_data["image_path"] = f"{document_name}/image{i}"
                        self.s3_handler.upload_image_text(document_name, image_text[i], i)
                        # Base64 to BytesIO binary
                        binary_image = BytesIO(base64.b64decode(image.attachment_data)) # attachment_data is already the base64 string
                        try:
                            # Convert BytesIO to PIL Image for embedding
                            embedding_ready_image = Image.open(binary_image)
                            embedded_image = self.embedder.image_to_embedding(embedding_ready_image)
                            embedding_ready_image.close()
                            
                            # Reset BytesIO position for S3 upload
                            binary_image.seek(0)
                            image_s3_uri = self.s3_handler.upload_image(document_name, binary_image, i)
                            
                            self.vector_database.add_data(embedded_image, {
                                "document_path": f"s3://{S3Bucket.DOCUMENTS.value}/{additional_data['key']}",
                                "graph_path": f"s3://{S3Bucket.GRAPHS.value}/{graph_path}",
                                "image_path": image_s3_uri,
                                "type": "attachment_image",
                            })
                            logger.debug(f"Image {i} uploaded to S3 and added to vector database")
                            appended_data += f"Image {i} from Page {image.additional_data['page']}: {image_text[i]}\n"
                        finally:
                            binary_image.close()
                    if data is not None:
                        chunked_data = self.embedder.chunk_text(data)
                        for chunk in chunked_data:
                            embedded_chunk = self.embedder.text_to_embedding(chunk)
                            self.vector_database.add_data(embedded_chunk, {
                                "document_path": f"s3://{S3Bucket.DOCUMENTS.value}/{additional_data['key']}",
                                "graph_path": f"s3://{S3Bucket.GRAPHS.value}/{graph_path}",
                                "text": chunk,
                                "type": "text"
                            })
                        
                        data += appended_data
                        self.s3_handler.upload_document_text(document_name, data, file_type="summary")
                        graph = self.graphModel.create_graph_dict(data)
                        self.s3_handler.upload_graph(document_name, json.dumps(graph))
            else:
                raise ValueError("Unknown attachment data type" + str(type(attachment_file.attachment_data)))
        logger.debug("Exiting process_file")

    def process_files(self, files):
        logger.debug("Entering process_files with files=%s", files)
        for file in files:
            # We download the file and go through our temp_download s3 process
            self.s3_handler.download_to_temp_and_process(
                bucket=S3Bucket.DOCUMENTS,
                key=file,
                process_callback=self.process_file
            )
        logger.debug("Exiting process_files")