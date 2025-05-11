import yaml
import tempfile
import os
from psycore import Psycore
from typing import Optional, Dict, Any, Union


class PsycoreTestRunner:
    def __init__(self, config: Optional[Union[str, Dict[str, Any]]] = None, preprocess: bool = True):
        """
        Initialize the PsycoreRunner with optional configuration.
        
        Args:
            config: Either a YAML string or a dictionary containing the configuration
            preprocess: Whether to run preprocessing during initialization
        """
        self.config = self._get_default_config() if config is None else config
        self.psycore = None
        
        self.resetPsycore()
        if preprocess:
            self.preprocess()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return the default configuration."""
        return {
            "model": {
                "primary": "oai_4o_latest",
                "allow_image_input": True
            },
            "graph_verification": {
                "enabled": True,
                "method": "llm",
                "llm_model": "oai_4o_latest"
            },
            "prompt_mode": {
                "mode": "elaborated",
                "elaborator_model": "oai_4o_latest"
            },
            "text_summariser": {
                "model": "oai_4o_latest"
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
                "end_index": 3
            },
            "rag": {
                "text_similarity_threshold": 0.55
            }
        }
    
    def update_config(self, updates: Dict[str, Any], preprocess: bool = True) -> 'PsycoreTestRunner':
        """
        Update specific configuration parameters.
        
        Args:
            updates: Dictionary containing the configuration updates
            preprocess: Whether to run preprocessing after updating config
            
        Returns:
            self for method chaining
        """
        def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = deep_update(d[k], v)
                else:
                    d[k] = v
            return d
        
        self.config = deep_update(self.config, updates)
        self.resetPsycore()
        if preprocess:
            self.preprocess()
            
        return self
    
    def _create_temp_config_file(self) -> str:
        """Create a temporary YAML file with the current configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
            yaml.dump(self.config, temp_file)
            return temp_file.name
        
    def resetPsycore(self) -> 'PsycoreTestRunner':
        """
        Reset the Psycore instance.
        
        Returns:
            self for method chaining
        """
        try:
            temp_file_path = self._create_temp_config_file()
            self.psycore = Psycore(temp_file_path)
        finally:
            os.unlink(temp_file_path)
    


        return self
    
    def preprocess(self) -> 'PsycoreTestRunner':
        """
        Run preprocessing with the current configuration.
        
        Returns:
            self for method chaining
        """
        self.psycore.preprocess(skip_confirmation=True)
        
    def evaluate_prompt(self, prompt: str) -> Any:
        """
        Evaluate a prompt using the current configuration.
        
        Args:
            prompt: The prompt to evaluate
            
        Returns:
            The evaluation result
        """
        if self.psycore is None:
            raise RuntimeError("Psycore instance not initialized. Call preprocess() first or initialize with preprocess=True")
        return self.psycore.evaluate_prompt(prompt)
    
    def process_evaluation(self, evaluation: dict) -> dict:
        """
        Process the prompt evaluation result.

        Args:
            evaluation: The evaluation result
            
        Returns:
            The processed evaluation result
        """
        # This function is TODO but it's to unify the output as a general manner between all the evaluators and the various documents the result returned
        return evaluation
    

    def evaluate_prompts(self, prompts: list[str]) -> list[Any]:
        """
        Evaluate a list of prompts using the current configuration.
        
        Args:
            prompts: List of prompts to evaluate
            
        """
        return [self.evaluate_prompt(prompt) for prompt in prompts]


# Example usage
if __name__ == "__main__":
    # Create a runner with default config and run preprocessing
    runner = PsycoreTestRunner(preprocess=True)
    
    # Example of updating specific parameters and re-preprocessing
    runner.update_config({
        "model": {
            "primary": "oai_4o_latest"
        },
        "document_range": {
            "start_index": 0,
            "end_index": 3
        }
    })
    
    prompts = ["What is the capital of France?", "What is the capital of Germany?"]
    results = runner.evaluate_prompts(prompts)