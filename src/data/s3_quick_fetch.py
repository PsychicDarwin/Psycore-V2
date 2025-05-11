from .s3_handler import S3Handler, S3Bucket
from .attachments import Attachment, AttachmentTypes
class S3QuickFetch:
    def __init__(self, s3_handler: S3Handler):
        self.s3_handler = s3_handler

    def get_image(self, image_s3: str) -> Attachment:
        local_path = self.s3_handler.temp_download_file(image_s3)
        attachment = Attachment(attachment_type=AttachmentTypes.IMAGE, attachment_data=local_path,needs_extraction=True)
        attachment.extract()
        self.s3_handler.cleanup_temp_file(local_path)
        return attachment


    def fetch_text(self, text_s3: str) -> str:
        file = self.s3_handler.temp_download_file(text_s3)
        file_data = open(file, "r").read()
        self.s3_handler.cleanup_temp_file(file)
        return file_data
