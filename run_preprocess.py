import yaml
import tempfile
import os
from psycore import Psycore
from typing import Optional, Dict, Any, Union


class PsycoreRunner:
    def __init__(self, config: Optional[Union[str, Dict[str, Any]]] = None):
        """
        Initialize the PsycoreRunner with optional configuration.
        
        Args:
            config: Either a YAML string or a dictionary containing the configuration
        """
        self.config = self._get_default_config() if config is None else config
        self.psycore = None
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Return the default configuration."""
        return {
            "model": {
                "primary": "oai_4o_latest",
                "allow_image_input": True
            },
            "graph_verification": {
                "enabled": True,
                "method": "bert",
                "llm_model": "oai_4o_latest"
            },
            "prompt_mode": {
                "mode": "elaborated",
                "elaborator_model": "oai_4o_latest"
            },
            "text_summariser": {
                "model": "llava_13b"
            },
            "embedding": {
                "method": "aws",
                "model": "amazon.titan-embed-image-v1"
            },
            "logger": {
                "level": "DEBUG"
            },
            "document_range": {
                "enabled": True,
                "start_index": 0,
                "end_index": 10
            },
            "rag": {
                "text_similarity_threshold": 0.55
            }
        }
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update specific configuration parameters.
        
        Args:
            updates: Dictionary containing the configuration updates
        """
        def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = deep_update(d[k], v)
                else:
                    d[k] = v
            return d
        
        self.config = deep_update(self.config, updates)
    
    def _create_temp_config_file(self) -> str:
        """Create a temporary YAML file with the current configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
            yaml.dump(self.config, temp_file)
            return temp_file.name
    
    def preprocess(self, skip_confirmation: bool = True) -> 'PsycoreRunner':
        """
        Run preprocessing with the current configuration.
        
        Args:
            skip_confirmation: Whether to skip the confirmation prompt
            
        Returns:
            self for method chaining
        """
        temp_file_path = self._create_temp_config_file()
        try:
            self.psycore = Psycore(temp_file_path)
            self.psycore.preprocess(skip_confirmation=skip_confirmation)
            return self
        finally:
            os.unlink(temp_file_path)
    
    def evaluate_prompt(self, prompt: str, preprocess: bool = True) -> Any:
        """
        Evaluate a prompt using the current configuration.
        
        Args:
            prompt: The prompt to evaluate
            preprocess: Whether to run preprocessing before evaluation
            
        Returns:
            The evaluation result
        """
        if preprocess or self.psycore is None:
            self.preprocess()
        return self.psycore.evaluate_prompt(prompt)


# Example usage
if __name__ == "__main__":
    # Create a runner with default config
    runner = PsycoreRunner()
    
    # Example of updating specific parameters
    runner.update_config({
        "model": {
            "primary": "oai_4o_latest"
        },
        "document_range": {
            "start_index": 0,
            "end_index": 5
        }
    })
    
    # Run evaluation
    result = runner.evaluate_prompt("What is the capital of France?", preprocess=True)