import yaml
import tempfile
import os
from psycore import Psycore


def run_preprocess_with_config(config_yaml_str, skip_confirmation=True):
    """
    Run preprocessing with a custom YAML configuration string.
    
    Args:
        config_yaml_str (str): YAML configuration as a string
        skip_confirmation (bool): Whether to skip the confirmation prompt
    
    Returns:
        Psycore: The initialized Psycore instance
    """
    # Create a temporary file to store the YAML config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        temp_file.write(config_yaml_str)
        temp_file_path = temp_file.name
    
    try:
        # Initialize Psycore with the temporary config file
        psycore = Psycore(temp_file_path)
        
        # Run preprocessing
        psycore.preprocess(skip_confirmation=skip_confirmation)
        
        return psycore
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)


def run_test(config_yaml_str, prompt, preprocess=True):
    if preprocess:
        psycore = run_preprocess_with_config(config_yaml_str, skip_confirmation=True)
    return psycore.evaluate_prompt(prompt)

if __name__ == "__main__":
    # Example usage
    example_config = """
model:
  primary: "oai_4o_latest"
  allow_image_input: true

graph_verification:
  enabled: true
  method: "bert"
  llm_model: "oai_4o_latest"

prompt_mode:
  mode: "elaborated"
  elaborator_model: "oai_4o_latest"

text_summariser:
  model: "llava_13b"

embedding:
  method: "aws"
  model: "amazon.titan-embed-image-v1"

logger:
  level: "DEBUG"

document_range:
  enabled: true
  start_index: 0
  end_index: 10

rag:
  text_similarity_threshold: 0.55
"""
    
    # Run preprocessing with the example config
    run_test(example_config, "What is the capital of France?",False)