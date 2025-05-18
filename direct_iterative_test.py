"""
Simplified test for Psycore-V2 iterative stage.
This script focuses only on testing the iterative stage component directly.
"""

import sys
import os
import json
import traceback
from src.system_manager import LoggerController, ConfigManager, LocalCredentials
from src.data.s3_handler import S3Handler, S3Bucket
from src.data.s3_quick_fetch import S3QuickFetch
from src.kg import BERT_KG
from src.llm.wrappers import ChatModelWrapper
from src.llm import ModelCatalogue
from src.main.iterative_stage import IterativeStage

def test_iterative_stage_directly():
    """Test the iterative stage component directly without using PsycoreTestRunner"""
    
    # Set up logging
    LoggerController.initialize("DEBUG")
    logger = LoggerController.get_logger()
    
    logger.info("Starting direct iterative stage test")
    
    try:
        # Step 1: Initialize S3 handler
        logger.info("Initializing S3 handler...")
        s3_creds = {
            "aws_iam": LocalCredentials.get_credential('AWS_IAM_KEY'),
            "region": LocalCredentials.get_credential('AWS_DEFAULT_REGION').secret_key,
            "buckets": {
                "documents": LocalCredentials.get_credential('S3_DOCUMENTS_BUCKET').secret_key,
                "text": LocalCredentials.get_credential('S3_TEXT_BUCKET').secret_key,
                "images": LocalCredentials.get_credential('S3_IMAGES_BUCKET').secret_key,
                "graphs": LocalCredentials.get_credential('S3_GRAPHS_BUCKET').secret_key
            }
        }
        s3_handler = S3Handler(s3_creds)
        s3_quick_fetch = S3QuickFetch(s3_handler)
        
        # Step 2: Initialize BERT knowledge graph model
        logger.info("Initializing BERT knowledge graph model...")
        graph_model = BERT_KG()
        
        # Step 3: Get some sample RAG results
        # Note: Normally these would come from running RAG,
        # but for direct testing we'll create mock results
        logger.info("Creating sample RAG results...")
        
        # First try to get actual files from S3 to use as examples
        try:
            documents = s3_handler.list_base_directory_files(S3Bucket.DOCUMENTS)
            logger.info(f"Found {len(documents)} documents in S3")
            
            if len(documents) > 0:
                # Use the first document as an example
                document_path = documents[0]
                
                # Try to find corresponding graph file
                graphs = s3_handler.list_base_directory_files(S3Bucket.GRAPHS)
                graph_path = None
                
                for graph in graphs:
                    if document_path.split('/')[-1] in graph:
                        graph_path = graph
                        break
                
                if graph_path:
                    logger.info(f"Using document: {document_path} and graph: {graph_path}")
                    
                    # Create a simple mock RAG result
                    rag_results = [{
                        "vector_id": "mock_vector_id_1",
                        "score": 0.85,
                        "document_path": document_path,
                        "graph_path": graph_path,
                        "type": "text",
                        "text": "This is sample text for testing the iterative stage."
                    }]
                else:
                    logger.warning("No matching graph found, using mock RAG results")
                    rag_results = create_mock_rag_results()
            else:
                logger.warning("No documents found in S3, using mock RAG results")
                rag_results = create_mock_rag_results()
                
        except Exception as e:
            logger.error(f"Error accessing S3: {str(e)}")
            logger.info("Using mock RAG results instead")
            rag_results = create_mock_rag_results()
        
        # Step 4: Create a mock chat result
        class MockChatResult:
            def __init__(self, content):
                self.content = content
        
        rag_chat_results = MockChatResult(
            "This is a test response that should be analyzed by the iterative stage. "
            "It contains information that may or may not match the knowledge graph."
        )
        
        # Step 5: Initialize and test the iterative stage
        logger.info("Initializing iterative stage...")
        threshold = 0.7  # High threshold to encourage iterations
        iterative_stage = IterativeStage(s3_quick_fetch, graph_model, threshold, rag_results)
        
        # Step 6: Run the decision maker
        logger.info("Running iterative stage decision maker...")
        decision_result = iterative_stage.decision_maker(rag_results, rag_chat_results)
        
        # Step 7: Analyze and print results
        print("\n=== Iterative Stage Test Results ===\n")
        print(f"Valid relations percentage: {decision_result[0]}")
        print(f"Pass threshold: {threshold}")
        print(f"Passed validation: {decision_result[0] >= threshold}")
        
        missing_relations = decision_result[1]
        print(f"\nNumber of missing relations: {len(missing_relations)}")
        
        if len(missing_relations) > 0:
            print("\nSample missing relations:")
            for i, relation in enumerate(missing_relations[:5]):
                print(f"  {i+1}. {relation}")
                if i >= 4:
                    break
        
        llm_graph = decision_result[2]
        print(f"\nTotal relations in LLM response: {len(llm_graph)}")
        
        logger.info("Direct iterative stage test completed")
        
        return decision_result
        
    except Exception as e:
        logger.error(f"Error in direct iterative stage test: {str(e)}")
        traceback.print_exc()
        print(f"\nError: {str(e)}")
        print("\nCheck the logs for more details.")
        return None

def create_mock_rag_results():
    """Create mock RAG results for testing when S3 is not available"""
    return [{
        "vector_id": "mock_vector_id_1",
        "score": 0.85,
        "document_path": "mock_document_path.pdf",
        "graph_path": "mock_graph_path.json",
        "type": "text",
        "text": "This is sample text for testing the iterative stage."
    }]

if __name__ == "__main__":
    print("=== Direct Iterative Stage Test ===")
    result = test_iterative_stage_directly()
    
    if result:
        print("\nTest completed successfully!")
        print("You can now analyze the results above to understand how the iterative stage works.")
    else:
        print("\nTest failed. Please check the logs for more details.")
