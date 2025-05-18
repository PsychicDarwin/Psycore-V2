"""
Test script for Psycore-V2 iterative stage.
This script provides a simple way to test the iterative stage functionality.
"""

import sys
import os
import json
import traceback
from psycore import Psycore
from PsycoreTestRunner import PsycoreTestRunner
from src.system_manager import LoggerController

def main():
    # Set up logging
    LoggerController.initialize("DEBUG")
    logger = LoggerController.get_logger()
    
    logger.info("Starting Psycore iterative stage test")
    
    try:
        # Step 1: Create a test runner with a specific configuration
        logger.info("Creating PsycoreTestRunner...")
        runner = PsycoreTestRunner(
            config={
                "model": {
                    "primary": "oai_4o_latest",  # Using OpenAI's latest model
                    "allow_image_input": True
                },
                "graph_verification": {
                    "enabled": True,  # Enable graph verification for iterative stage
                    "method": "bert",  # Using BERT for faster testing
                },
                "prompt_mode": {
                    "mode": "elaborated",
                    "elaborator_model": "oai_4o_latest"
                },
                "text_summariser": {
                    "model": "llava_13b"  # Using llava for image processing
                },
                "embedding": {
                    "method": "aws",
                    "model": "amazon.titan-embed-image-v1"  # AWS embedding model
                },
                "document_range": {
                    "enabled": True,
                    "document_ids": [0, 1]  # Using first two documents for testing
                },
                "iteration": {
                    "loop_retries": 3,  # Maximum number of iterations
                    "pass_threshold": 0.7  # Higher threshold to encourage iterations
                },
                "rag": {
                    "text_similarity_threshold": 0.55
                }
            },
            preprocess=True  # Run preprocessing during initialization
        )
        
        logger.info("PsycoreTestRunner created successfully")
        
        # Step 2: Define a test prompt
        test_prompt = "Explain all the connections between the components in the technical documentation"
        logger.info(f"Testing prompt: {test_prompt}")
        
        # Step 3: Run the evaluation
        logger.info("Running evaluation...")
        evaluation_results = runner.evaluate_prompt(test_prompt)
        logger.info("Evaluation complete!")
        
        # Step 4: Save the results to a file
        output_file = "iterative_test_results.json"
        with open(output_file, "w") as f:
            json.dump(evaluation_results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
        
        # Step 5: Print a summary of the results
        print("\n==== Iterative Test Results ====\n")
        print(f"Prompt: {test_prompt}")
        print(f"Number of documents retrieved: {len(evaluation_results)}")
        
        # Step 6: Examine each document result
        for j, result in enumerate(evaluation_results):
            print(f"\n--- Document {j+1}: {result.get('document_path', 'Unknown')} ---")
            print(f"Initial score: {result.get('score', 'Unknown')}")
            
            # Look for graph evaluation data
            if 'graph_evaluation' in result:
                print(f"Graph evaluation score: {result.get('graph_evaluation', {}).get('score', 'Unknown')}")
                missing_relations = result.get('graph_evaluation', {}).get('missing_relations', [])
                valid_relations = result.get('graph_evaluation', {}).get('valid_relations', [])
                print(f"Missing relations: {len(missing_relations)}")
                print(f"Valid relations: {len(valid_relations)}")
        
        logger.info("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        traceback.print_exc()
        print(f"\nError: {str(e)}")
        print("\nCheck the logs for more details.")

if __name__ == "__main__":
    main()
