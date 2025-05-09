import os
import uuid
import boto3
from typing import Dict, List, Any, BinaryIO, Optional, Tuple, Union
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import tempfile
from src.system_manager import LocalCredentials
from enum import Enum

# Load environment variables
load_dotenv()

class S3Bucket(Enum):
    DOCUMENTS = LocalCredentials.get_credential('S3_DOCUMENTS_BUCKET').secret_key
    TEXT = LocalCredentials.get_credential('S3_TEXT_BUCKET').secret_key
    IMAGES = LocalCredentials.get_credential('S3_IMAGES_BUCKET').secret_key
    GRAPHS = LocalCredentials.get_credential('S3_GRAPHS_BUCKET').secret_key


class S3Handler:
    def __init__(self, creds={}):
        aws_cred = creds["aws_iam"]
        session = boto3.Session(
            aws_access_key_id=aws_cred.user_key,
            aws_secret_access_key=aws_cred.secret_key,
            region_name=creds["region"]
        )
        self.account_id = session.client('sts').get_caller_identity().get('Account')
        self.region = session.region_name
        self.s3 = session.client('s3')

    def list_base_directory_files(self,  bucket: S3Bucket) -> List[str]:
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

    def _upload_to_s3(self,bucket: S3Bucket, key: str, body: Union[bytes, BinaryIO], content_type: str) -> str:
        try:
            if isinstance(body, bytes):
                self.s3.put_object(Bucket=bucket.value, Key=key, Body=body, ContentType=content_type)
            else:
                self.s3.upload_fileobj(body, bucket, key)
            return f"s3://{bucket.value}/{key}"
        except ClientError as e:
            print(f"Error uploading to {bucket.value}/{key}: {e}")
            raise

    def parse_s3_uri(self, s3_uri: str) -> tuple[str, str]:
        if not s3_uri.startswith('s3://'):
            raise ValueError(f"Invalid S3 URI format: {s3_uri}")
        parts = s3_uri[5:].split('/', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URI format: {s3_uri}")
        return parts[0], parts[1]

    def upload_document(self, file_obj: BinaryIO, original_filename: str) -> Tuple[str, str]:
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
        return document_id, s3_link

    def upload_document_text(self, doc_s3_link: str, text_content: str, file_type: str = "main") -> str:
        doc_id = doc_s3_link.split("/")[-1].split(".")[0]
        key = f"{doc_id}/{file_type}.txt"
        return self._upload_to_s3(S3Bucket.TEXT.value, key, text_content.encode('utf-8'), 'text/plain')

    def upload_document_summary(self, document_id: str, summary_content: str) -> str:
        return self.upload_document_text(document_id, summary_content, file_type="summary")

    def upload_image(self, document_id: str, image_data: BinaryIO, image_number: int, extension: str = ".png") -> str:
        key = f"{document_id}/image{image_number}{extension}"
        return self._upload_to_s3(S3Bucket.IMAGES.value, key, image_data, 'image/png')

    def upload_image_text(self, document_id: str, text_content: str, image_number: int) -> str:
        return self.upload_document_text(document_id, text_content, file_type=f"image{image_number}")

    def upload_graph(self, document_id: str, graph_json: str) -> str:
        key = f"{document_id}/graph.json"
        return self._upload_to_s3(S3Bucket.GRAPHS.value, key, graph_json.encode('utf-8'), 'application/json')

    def concat_and_replace_summary(self, document_id: str) -> str:
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
        
    def download_file(self, s3_link: str) -> bytes:
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

    def download_text(self, s3_link: str) -> str:
        """
        Download and decode a text file from S3.
        """
        binary_content = self.download_file(s3_link)
        return binary_content.decode('utf-8')

    def download_to_temp_and_process(self, bucket: S3Bucket, key: str, process_callback, file_extension: Optional[str] = None) -> Any:
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

    def process_s3_file(self, file_info: Dict[str, str], process_callback) -> Any:
        """
        Process an S3 file using metadata and a callback.
        """
        key = file_info['Key']
        bucket = file_info['Bucket']
        _, file_extension = os.path.splitext(key)
        return self.download_to_temp_and_process(bucket, key, process_callback, file_extension)
    

    def reset_buckets(self) -> None:
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