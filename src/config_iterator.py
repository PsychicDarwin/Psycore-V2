import yaml
import itertools
from typing import Dict, List, Any, Tuple
import os
from pathlib import Path
from src.llm.model_catalogue import ModelCatalogue
import hashlib

class ConfigIterator:
    # Static model lists
    MAX_DOWNLOAD_SIZE = 14
    ALL_MODELS = list(ModelCatalogue.get_best_in_family(ModelCatalogue.get_testing_models(ModelCatalogue.get_api_models())).keys())
    MULTIMODAL_MODELS = list(ModelCatalogue.get_testing_models(ModelCatalogue.get_api_models(
        ModelCatalogue.get_MLLMs()
    )).keys())
    JSON_FRIENDLY_MODELS = list(ModelCatalogue.get_testing_models(
        ModelCatalogue.get_models_with_json_schema(ModelCatalogue.get_api_models()
    )).keys())
    
    # Define model categories
    API_LIMITED_MODELS = {
        'mistral_24.02_large',
        'claude_3_sonnet',
        'meta_llama_3_70b_instruct'
    }

    HARDWARE_INTENSIVE_MODELS = {
        'llava',
        'bakava',
        'deepseek',
        'bert',
        'clip',
        'llava_7b',
        'deepseek_1.5b_r1',
        'bakllava_7b'
    }
    
    # Static base variations
    BASE_VARIATIONS = {
        "model.primary": MULTIMODAL_MODELS,  # Only use multimodal models for primary
        "prompt_mode.elaborator_model": ALL_MODELS,  # Use any model for elaborator
        "text_summariser.model": MULTIMODAL_MODELS,  # Only use multimodal models for text summarizer
        "prompt_mode.mode": ["original", "elaborated"],
        "rag.text_similarity_threshold": [0.3, 0.7],
        "iteration.pass_threshold": [0.5, 0.7, 0.9],  # Added pass threshold variations
        "iteration.loop_retries": [3, 5, 7],  # Will be ignored if pass_threshold is 0
        "embedding.method": ["clip", "aws"],
        "graph_verification.method": ["llm", "bert"]
    }
    
    def __init__(self, base_config_path: str = "../config.yaml"):
        """Initialize the config iterator with a base configuration file."""
        self.base_config_path = base_config_path
        self.base_config = self._load_config(base_config_path)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load a YAML configuration file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _hash_config(self, config: Dict) -> str:
        """Generate a hash of the config content."""
        config_str = yaml.dump(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def _get_model_category(self, config: Dict) -> str:
        """Determine the category of models used in the config."""
        models_used = set()
        
        # Check primary model
        if "model" in config and "primary" in config["model"]:
            primary_model = config["model"]["primary"].lower()
            models_used.add(primary_model)
            print(f"Found primary model: {primary_model}")  # Debug print
        
        # Check elaborator model
        if "prompt_mode" in config:
            # Check elaborator model if it exists
            if "elaborator_model" in config["prompt_mode"]:
                elaborator_model = config["prompt_mode"]["elaborator_model"].lower()
                models_used.add(elaborator_model)
                print(f"Found elaborator model: {elaborator_model}")  # Debug print
            # If mode is not 'original', we should use the elaborator model
            elif "mode" in config["prompt_mode"] and config["prompt_mode"]["mode"] != "original":
                # Use the primary model as elaborator if no specific elaborator is set
                if "model" in config and "primary" in config["model"]:
                    primary_model = config["model"]["primary"].lower()
                    models_used.add(primary_model)
                    print(f"Using primary model as elaborator: {primary_model}")  # Debug print
        
        # Check text summarizer
        if "text_summariser" in config and "model" in config["text_summariser"]:
            summarizer_model = config["text_summariser"]["model"].lower()
            models_used.add(summarizer_model)
            print(f"Found summarizer model: {summarizer_model}")  # Debug print
        
        # Check graph LLM
        if "graph_verification" in config and "llm_model" in config["graph_verification"]:
            graph_llm = config["graph_verification"]["llm_model"].lower()
            models_used.add(graph_llm)
            print(f"Found graph LLM: {graph_llm}")  # Debug print
        
        # Add graph method if it's a model
        if "graph_verification" in config and "method" in config["graph_verification"]:
            method = config["graph_verification"]["method"].lower()
            if method in ["llm", "bert"]:
                models_used.add(method)
                print(f"Found graph method: {method}")  # Debug print
        
        # Add embedding method if it's a model
        if "embedding" in config and "method" in config["embedding"]:
            method = config["embedding"]["method"].lower()
            if method in ["clip"]:
                models_used.add(method)
                print(f"Found embedding method: {method}")  # Debug print
        
        print(f"All models used: {models_used}")  # Debug print
        
        has_api_limited = any(model in self.API_LIMITED_MODELS for model in models_used)
        has_hardware_intensive = any(model in self.HARDWARE_INTENSIVE_MODELS for model in models_used)
        
        print(f"Has API limited: {has_api_limited}")  # Debug print
        print(f"Has hardware intensive: {has_hardware_intensive}")  # Debug print
        
        if has_hardware_intensive:
            return 'API_and_Hardware_Intensive'
        elif has_api_limited:
            return 'API_Limited'
        else:
            return 'General_Models'

    def _get_folder_structure(self, config: Dict) -> Tuple[str, str]:
        """
        Determine the folder structure based on config variations.
        Returns a tuple of (category_folder, method_folder)
        """
        # Get the category folder
        category = self._get_model_category(config)
        
        # Get the method folder
        graph_method = config.get('graph_verification', {}).get('method', '').lower()
        embedding_method = config.get('embedding', {}).get('method', '').lower()
        
        method_folder = None
        if graph_method == 'bert' and embedding_method == 'clip':
            method_folder = 'BERT_Graph_CLIP_Embedding'
        elif graph_method == 'bert' and embedding_method == 'aws':
            method_folder = 'BERT_Graph_AWS_Embedding'
        elif graph_method == 'llm' and embedding_method == 'clip':
            method_folder = 'LLM_Graph_CLIP_Embedding'
        elif graph_method == 'llm' and embedding_method == 'aws':
            method_folder = 'LLM_Graph_AWS_Embedding'
        else:
            method_folder = 'Unknown_Method'
            
        return category, method_folder

    def _save_config(self, config: Dict, output_path: str, written_hashes: Dict[str, str]) -> bool:
        """
        Save a configuration to a YAML file if it hasn't been written before.
        
        Args:
            config: The configuration to save
            output_path: Path where to save the config
            written_hashes: Dictionary tracking hashes of written configs
            
        Returns:
            bool: True if config was written, False if it was a duplicate
        """
        config_hash = self._hash_config(config)
        if config_hash in written_hashes:
            print(f"Skipping duplicate config (matches {written_hashes[config_hash]})")
            return False
            
        # Create directory structure
        category, method_folder = self._get_folder_structure(config)
        full_path = os.path.join(
            os.path.dirname(output_path),
            category,
            method_folder
        )
        os.makedirs(full_path, exist_ok=True)
        
        # Update output path to include new directory structure
        new_output_path = os.path.join(full_path, os.path.basename(output_path))
        
        with open(new_output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        written_hashes[config_hash] = new_output_path
        return True

    def _handle_original_mode(self, new_config: Dict) -> None:
        """Handle the case where prompt_mode.mode is 'original' by reverting elaborator_model."""
        try:
            if (
                'prompt_mode' in new_config and
                'mode' in new_config['prompt_mode'] and
                new_config['prompt_mode']['mode'] == 'original'
            ):
                base_elab = self.base_config.get('prompt_mode', {}).get('elaborator_model', None)
                if base_elab is not None:
                    new_config['prompt_mode']['elaborator_model'] = base_elab
                elif 'elaborator_model' in new_config['prompt_mode']:
                    del new_config['prompt_mode']['elaborator_model']
        except Exception as e:
            print(f"Warning: Could not check/revert elaborator_model: {e}")

    def _handle_embedding_method(self, new_config: Dict) -> None:
        """Handle embedding method-specific configurations."""
        if "embedding.method" not in new_config:
            return

        method = new_config["embedding"]["method"]
        if method == "aws":
            new_config["embedding"]["model"] = "amazon.titan-embed-image-v1"
        elif method == "clip" and "model" in new_config["embedding"]:
            del new_config["embedding"]["model"]

    def _handle_graph_verification(self, new_config: Dict, output_dir: str, config_count: int, written_hashes: Dict[str, str]) -> int:
        """Handle graph verification method-specific configurations and save configs."""
        if "graph_verification.method" not in new_config:
            config_count += 1
            output_path = os.path.join(output_dir, f"config_variation_{config_count}.yaml")
            if self._save_config(new_config, output_path, written_hashes):
                print(f"Generated config variation {config_count}")
            return config_count

        method = new_config["graph_verification"]["method"]
        if method == "llm":
            for json_model in self.JSON_FRIENDLY_MODELS:
                llm_config = self._deep_copy_config(new_config)
                llm_config["graph_verification"]["llm_model"] = json_model
                config_count += 1
                output_path = os.path.join(output_dir, f"config_variation_{config_count}.yaml")
                if self._save_config(llm_config, output_path, written_hashes):
                    print(f"Generated config variation {config_count}")
        elif method == "bert":
            if "llm_model" in new_config["graph_verification"]:
                del new_config["graph_verification"]["llm_model"]
            config_count += 1
            output_path = os.path.join(output_dir, f"config_variation_{config_count}.yaml")
            if self._save_config(new_config, output_path, written_hashes):
                print(f"Generated config variation {config_count}")

        return config_count

    def get_all_variations(self, variations: Dict[str, List[Any]], output_dir: str = "config_variations") -> int:
        """
        Generate all possible configuration variations and write them to files as they are produced.
        
        Args:
            variations: Dictionary mapping config paths to lists of possible values
                      e.g., {"model.primary": ["oai_4o_latest", "gpt-3.5-turbo"]}
            output_dir: Directory to save the generated config files
        
        Returns:
            Number of configurations generated
        """
        os.makedirs(output_dir, exist_ok=True)
        
        keys = list(variations.keys())
        values = list(variations.values())
        combinations = list(itertools.product(*values))
        
        config_count = 0
        written_hashes = {}  # Track hashes of written configs
        
        for combination in combinations:
            new_config = self._deep_copy_config(self.base_config)
            
            # Apply the variation
            for key, value in zip(keys, combination):
                self._set_nested_value(new_config, key.split('.'), value)

            # Handle special cases
            self._handle_original_mode(new_config)
            self._handle_embedding_method(new_config)
            config_count = self._handle_graph_verification(new_config, output_dir, config_count, written_hashes)
        
        return config_count
    
    def _deep_copy_config(self, config: Dict) -> Dict:
        """Create a deep copy of the configuration dictionary."""
        return yaml.safe_load(yaml.dump(config))
    
    def _set_nested_value(self, config: Dict, key_path: List[str], value: Any):
        """Set a value in a nested dictionary using a list of keys."""
        current = config
        for key in key_path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[key_path[-1]] = value

def main():
    # Example usage
    iterator = ConfigIterator()
    
    # Generate variations using the static BASE_VARIATIONS
    total_configs = iterator.get_all_variations(ConfigIterator.BASE_VARIATIONS)
    print(f"Generated {total_configs} total variations")

if __name__ == "__main__":
    main() 