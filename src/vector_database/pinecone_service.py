from pinecone import Pinecone, ServerlessSpec
import uuid
from src.system_manager import LoggerController
from src.vector_database.vector_service import VectorService
from numpy import ndarray

# Configure logging
logger = LoggerController.get_logger()

class PineconeService(VectorService):
    def __init__(self, embedder, credentials: dict):
        super().__init__(embedder)
        logger.info(f"Initializing PineconeService with index name: {credentials['index_name']}")
        self.index_name = credentials['index_name']
        self.service = Pinecone(
            api_key=credentials['api_key']
        )
        if self.index_name not in [i['name'] for i in self.service.list_indexes()]:
            logger.info(f"Creating new Pinecone index: {self.index_name}")
            self.service.create_index(
                name=self.index_name,
                dimension=embedder.dimension_output,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=credentials['aws_region']
                )
            )
        self.index = self.service.Index(self.index_name)
        logger.debug("PineconeService initialization complete")

    def gen_uuid(self):
        """Generate a unique ID for the data."""
        logger.debug("Generating new UUID for data")
        while True:
            uuid_id = str(uuid.uuid4())
            # Check if the ID exists in the index
            fetch_result = self.index.fetch(ids=[uuid_id])
            if not fetch_result.vectors:  # FetchResponse has a vectors attribute
                logger.debug(f"Generated unique UUID: {uuid_id}")
                return uuid_id
            logger.debug(f"UUID {uuid_id} already exists, generating new one")

    def add_data(self, embedding : ndarray, data: dict):
        """Add data to the vector database."""
        # Convert numpy array to list if necessary
        if isinstance(embedding, ndarray):
            embedding = embedding.tolist()
            
        uuid_id = self.gen_uuid()
        
        try:
            self.index.upsert(
                vectors=[{
                    'id': uuid_id,
                    'values': embedding,
                    'metadata': data
                }]
            )
            logger.info(f"Successfully added data with ID: {uuid_id}")
        except Exception as e:
            logger.error(f"Failed to add data to Pinecone: {str(e)}")
            raise

    def batch_add_data(self, embeddings: list[ndarray], data_list: list[dict], batch_size: int = 100):
        """Add multiple data points to the vector database in batches."""
        if len(embeddings) != len(data_list):
            raise ValueError("Number of embeddings must match number of data entries")
            
        # Convert numpy arrays to lists if necessary
        embeddings = [e.tolist() if isinstance(e, ndarray) else e for e in embeddings]
        
        # Process in batches
        for i in range(0, len(embeddings), batch_size):
            batch_embeddings = embeddings[i:i + batch_size]
            batch_data = data_list[i:i + batch_size]
            
            # Generate UUIDs for the batch
            batch_vectors = []
            for i, embedding, data in zip(range(len(batch_embeddings)), batch_embeddings, batch_data):
                uuid_id = self.gen_uuid()
                logger.info(f"Preparing chunk {i+1} of {len(batch_embeddings)}")
                batch_vectors.append({
                    'id': uuid_id,
                    'values': embedding,
                    'metadata': data
                })
            
            try:
                logger.info(f"Upserting batch of {len(batch_vectors)} vectors")
                self.index.upsert(vectors=batch_vectors)
                logger.info(f"Successfully added batch of {len(batch_vectors)} vectors")
            except Exception as e:
                logger.error(f"Failed to add batch to Pinecone: {str(e)}")
                raise

    def get_data(self, query: str,k=5) -> dict:
        """Get data from the vector database."""
        logger.info(f"Querying vector database with query: {query}, k={k}")
        embedding = self.embedder.text_to_embedding(query)
        results = self.index.query(
            vector=embedding.tolist(),
            top_k= k,
            include_metadata=True
        )
        return results['matches']

    def delete_data(self, data_id: str):
        """Delete data from the vector database."""
        logger.info(f"Deleting data with ID: {data_id}")
        self.index.delete(ids=[data_id])
        logger.info(f"Successfully deleted data with ID: {data_id}")

    def update_data(self, data_id: str, new_data: dict):
        """Update data in the vector database."""
        logger.info(f"Updating data with ID: {data_id}, new data: {new_data}")
        # Fetch the existing data
        existing_data = self.index.fetch(ids=[data_id])
        if existing_data:
            logger.debug(f"Found existing data for ID: {data_id}")
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
            logger.debug(f"Successfully updated data with ID: {data_id}")
        else:
            logger.warning(f"No existing data found for ID: {data_id}")

    def reset_data(self):
        """Reset the vector database."""
        print("Cleaning Pinecone index...")
        self.service.delete_index(self.index_name)
        logger.info(f"Deleted index: {self.index_name}")
        self.service.create_index(
            name=self.index_name,
            dimension=self.embedder.dimension_output,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        logger.info(f"Successfully recreated index: {self.index_name}")