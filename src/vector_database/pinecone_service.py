from pinecone import Pinecone, ServerlessSpec
import uuid
from src.vector_database.vector_service import VectorDatabaseService

class PineconeService(VectorDatabaseService):
    def __init__(self, embedder, credentials: dict):
        super().__init__(embedder)
        self.index_name = credentials['index_name']
        self.service = Pinecone(
            api_key=credentials['api_key']
        )

        if self.index_name not in self.service.list_indexes():
            self.service.create_index(
                name=self.index_name,
                dimension=embedder.chunk_size,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=credentials['aws_region']
                )
            )
        self.index = self.service.Index(self.index_name)

    def gen_uuid(self):
        """Generate a unique ID for the data."""
        id = str(uuid.uuid4())
        # Check if the ID already exists in the index
        while self.index.fetch(id):
            id = str(uuid.uuid4())
        return id

    def add_data(self, embedding : ndarray, data: dict):
        """Add data to the vector database."""
        id = self.gen_uuid()
        self.index.upsert(
            vectors=[{
                'id': id,
                'values': embedding,
                'metadata' : data
            }]
        )

    def get_data(self, query: str,k=5) -> dict:
        """Get data from the vector database."""
        embedding = self.embedder.text_to_embedding(query)
        results = self.index.query(
            vector=embedding,
            top_k= k,
            include_metadata=True
        )
        return results['matches']

    def delete_data(self, data_id: str):
        """Delete data from the vector database."""
        self.index.delete(ids=[data_id])

    def update_data(self, data_id: str, new_data: dict):
        """Update data in the vector database."""
        # Fetch the existing data
        existing_data = self.index.fetch(ids=[data_id])
        if existing_data:
            # Update the metadata
            existing_data['metadata'].update(new_data)
            # Upsert the updated data
            self.index.upsert(
                vectors=[{
                    'id': data_id,
                    'values': existing_data['values'],
                    'metadata': existing_data['metadata']
                }]
            )

    def reset_data(self):
        """Reset the vector database."""
        self.index.delete_all()
        self.index.create_index(
            name=self.index_name,
            dimension=self.embedder.chunk_size,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )