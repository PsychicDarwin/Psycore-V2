import os
import uuid
import boto3
from typing import Dict, List, Any, BinaryIO, Optional, Tuple, Union
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import tempfile
from src.system_manager import LocalCredentials
from enum import Enum
from src.system_manager import LoggerController

# Load environment variables
load_dotenv()

logger = LoggerController.get_logger()

class S3Bucket(Enum):
    DOCUMENTS = LocalCredentials.get_credential('S3_DOCUMENTS_BUCKET').secret_key
    TEXT = LocalCredentials.get_credential('S3_TEXT_BUCKET').secret_key
    IMAGES = LocalCredentials.get_credential('S3_IMAGES_BUCKET').secret_key
    GRAPHS = LocalCredentials.get_credential('S3_GRAPHS_BUCKET').secret_key


class S3Handler:
    def __init__(self, creds={}):
        logger.debug("Entering S3Handler.__init__")
        aws_cred = creds["aws_iam"]
        session = boto3.Session(
            aws_access_key_id=aws_cred.user_key,
            aws_secret_access_key=aws_cred.secret_key,
            region_name=creds["region"]
        )
        self.account_id = session.client('sts').get_caller_identity().get('Account')
        self.region = session.region_name
        self.s3 = session.client('s3')
        logger.debug("Exiting S3Handler.__init__")

    def list_base_directory_files(self,  bucket: S3Bucket) -> List[str]:
        logger.debug("Entering list_base_directory_files with bucket=%s", bucket)
        """
        Lists all files in the root of the specified S3 bucket (not recursively).
        """
        try:
            response = self.s3.list_objects_v2(Bucket=bucket.value, Prefix='', Delimiter='/')
            if 'Contents' not in response:
                return []
            return [obj['Key'] for obj in response['Contents'] if '/' not in obj['Key'].strip('/')]
        except ClientError as e:
            print(f"Error listing files in bucket {bucket.value}: {e}")
            raise
        logger.debug("Exiting list_base_directory_files")

    def _upload_to_s3(self,bucket: Union[S3Bucket, str], key: str, body: Union[bytes, BinaryIO, str], content_type: str) -> str:
        logger.debug("Entering _upload_to_s3 with bucket=%s, key=%s, content_type=%s", bucket, key, content_type)
        try:
            # Get bucket value if it's an S3Bucket enum
            bucket_name = bucket.value if isinstance(bucket, S3Bucket) else bucket
            
            if isinstance(body, (bytes, str)):
                # Convert string to bytes if needed
                if isinstance(body, str):
                    body = body.encode('utf-8')
                self.s3.put_object(Bucket=bucket_name, Key=key, Body=body, ContentType=content_type)
            else:
                self.s3.upload_fileobj(body, bucket_name, key)
            return f"s3://{bucket_name}/{key}"
        except ClientError as e:
            print(f"Error uploading to {bucket_name}/{key}: {e}")
            raise
        logger.debug("Exiting _upload_to_s3")

    def parse_s3_uri(self, s3_uri: str) -> tuple[str, str]:
        logger.debug("Entering parse_s3_uri with s3_uri=%s", s3_uri)
        if not s3_uri.startswith('s3://'):
            raise ValueError(f"Invalid S3 URI format: {s3_uri}")
        parts = s3_uri[5:].split('/', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URI format: {s3_uri}")
        return parts[0], parts[1]
        logger.debug("Exiting parse_s3_uri")

    def upload_document(self, file_obj: BinaryIO, original_filename: str) -> Tuple[str, str]:
        logger.debug("Entering upload_document with original_filename=%s", original_filename)
        base, extension = os.path.splitext(original_filename)
        key_base = f"documents/{base}"
        key = f"{key_base}{extension}"
        counter = 1

        while True:
            try:
                self.s3.head_object(Bucket=S3Bucket.DOCUMENTS.value, Key=key)
                key = f"{key_base}_{counter}{extension}"
                counter += 1
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    break
                else:
                    print(f"Error checking existence of {key}: {e}")
                    raise

        s3_link = self._upload_to_s3(S3Bucket.DOCUMENTS.value, key, file_obj, 'application/octet-stream')
        document_id = os.path.splitext(os.path.basename(key))[0]
        logger.debug("Exiting upload_document")
        return document_id, s3_link

    def upload_document_text(self, doc_s3_link: str, text_content: str, file_type: str = "main") -> str:
        logger.debug("Entering upload_document_text with doc_s3_link=%s, file_type=%s", doc_s3_link, file_type)
        doc_id = doc_s3_link.split("/")[-1].split(".")[0]
        key = f"{doc_id}/{file_type}.txt"
        return self._upload_to_s3(S3Bucket.TEXT, key, text_content, 'text/plain')
        logger.debug("Exiting upload_document_text")

    def upload_document_summary(self, document_id: str, summary_content: str) -> str:
        logger.debug("Entering upload_document_summary with document_id=%s", document_id)
        return self.upload_document_text(document_id, summary_content, file_type="summary")
        logger.debug("Exiting upload_document_summary")

    def upload_image(self, document_id: str, image_data: BinaryIO, image_number: int, extension: str = ".png") -> str:
        logger.debug("Entering upload_image with document_id=%s, image_number=%s, extension=%s", document_id, image_number, extension)
        key = f"{document_id}/image{image_number}{extension}"
        return self._upload_to_s3(S3Bucket.IMAGES, key, image_data, 'image/png')
        logger.debug("Exiting upload_image")

    def upload_image_text(self, document_id: str, text_content: str, image_number: int) -> str:
        logger.debug("Entering upload_image_text with document_id=%s, image_number=%s", document_id, image_number)
        return self.upload_document_text(document_id, text_content, file_type=f"image{image_number}")
        logger.debug("Exiting upload_image_text")

    def upload_graph(self, document_id: str, graph_json: str) -> str:
        logger.debug("Entering upload_graph with document_id=%s", document_id)
        key = f"{document_id}/graph.json"
        return self._upload_to_s3(S3Bucket.GRAPHS, key, graph_json.encode('utf-8'), 'application/json')
        logger.debug("Exiting upload_graph")

    def concat_and_replace_summary(self, document_id: str) -> str:
        logger.debug("Entering concat_and_replace_summary with document_id=%s", document_id)
        try:
            response = self.s3.list_objects_v2(Bucket=S3Bucket.TEXT.value, Prefix=f"{document_id}/")
            if 'Contents' not in response:
                raise ValueError(f"No text files found for document: {document_id}")

            texts = []
            for obj in response['Contents']:
                key = obj['Key']
                if key.endswith("summary.txt"):
                    continue
                text_obj = self.s3.get_object(Bucket=S3Bucket.TEXT.value, Key=key)
                text_content = text_obj['Body'].read().decode('utf-8')
                texts.append(text_content)

            full_summary = "\n".join(texts)
            return self.upload_document_summary(document_id, full_summary)
        except ClientError as e:
            print(f"Error concatenating summary: {e}")
            raise
        logger.debug("Exiting concat_and_replace_summary")

    def download_file(self, s3_link: str) -> bytes:
        logger.debug("Entering download_file with s3_link=%s", s3_link)
        """
        Download a file from S3 using its S3 URI.
        """
        bucket, key = self.parse_s3_uri(s3_link)
        try:
            response = self.s3.get_object(Bucket=bucket, Key=key)
            return response['Body'].read()
        except ClientError as e:
            print(f"Error downloading file: {e}")
            raise
        logger.debug("Exiting download_file")

    def download_text(self, s3_link: str) -> str:
        logger.debug("Entering download_text with s3_link=%s", s3_link)
        """
        Download and decode a text file from S3.
        """
        binary_content = self.download_file(s3_link)
        return binary_content.decode('utf-8')
        logger.debug("Exiting download_text")

    def download_to_temp_and_process(self, bucket: S3Bucket, key: str, process_callback, file_extension: Optional[str] = None) -> Any:
        logger.debug("Entering download_to_temp_and_process with bucket=%s, key=%s, file_extension=%s", bucket, key, file_extension)
        """
        Download a file to a temporary location, process it, and clean up.
        """
        if file_extension is None and '.' in key:
            _, file_extension = os.path.splitext(key)

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        local_path = temp_file.name
        temp_file.close()

        try:
            self.s3.download_file(bucket.value, key, local_path)
            extra_data = {'key': key, 'file_extension': file_extension}
            return process_callback(local_path, extra_data)
        except Exception as e:
            print(f"Error processing file {bucket.value}/{key}: {e}")
            raise
        finally:
            if os.path.exists(local_path):
                os.unlink(local_path)
        logger.debug("Exiting download_to_temp_and_process")

    def temp_download_file(self, s3_link: str, file_extension: Optional[str] = None) -> str:
        bucket, key = self.parse_s3_uri(s3_link)
        if file_extension is None and '.' in key:
            _, file_extension = os.path.splitext(key)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        local_path = temp_file.name
        temp_file.close()
        self.s3.download_file(bucket, key, local_path)
        return local_path
    
    # Not really an S3 method, but it's useful for the quick fetch class
    def cleanup_temp_file(self, local_path: str):
        if os.path.exists(local_path):
            os.unlink(local_path)

    def process_s3_file(self, file_info: Dict[str, str], process_callback) -> Any:
        logger.debug("Entering process_s3_file with file_info=%s", file_info)
        """
        Process an S3 file using metadata and a callback.
        """
        key = file_info['Key']
        bucket = file_info['Bucket']
        _, file_extension = os.path.splitext(key)
        return self.download_to_temp_and_process(bucket, key, process_callback, file_extension)
        logger.debug("Exiting process_s3_file")

    def reset_buckets(self) -> None:
        logger.debug("Entering reset_buckets")
        """
        Deletes all objects from the text, images, and graphs buckets.
        Use with caution.

        This will not reset the documents bucket.
        """
        buckets = [
            S3Bucket.TEXT.value,
            S3Bucket.IMAGES.value,
            S3Bucket.GRAPHS.value
        ]

        for bucket in buckets:
            try:
                response = self.s3.list_objects_v2(Bucket=bucket)
                if 'Contents' in response:
                    objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
                    # Boto3 supports batch deletion of up to 1000 objects
                    self.s3.delete_objects(
                        Bucket=bucket,
                        Delete={'Objects': objects_to_delete}
                    )
                    print(f"Cleared {len(objects_to_delete)} objects from {bucket}")
                else:
                    print(f"No objects found in {bucket}")
            except ClientError as e:
                print(f"Error resetting bucket {bucket}: {e}")
                raise
        logger.debug("Exiting reset_buckets")