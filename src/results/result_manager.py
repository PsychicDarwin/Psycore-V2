import pandas as pd
import os
import json
import hashlib
from typing import Dict, Any, List

class ResultManager:
    
    def __init__(self, directory: str = "./results", csv_locator: str = "results.csv", doc_id_delimiter: str = "|"):
        """
        Initializes the ResultManager with the given directory and CSV file name.

        :param directory: Directory where results and CSV will be stored
        :param csv_locator: Name of the CSV file for tracking results
        :param doc_id_delimiter: Delimiter to use when converting document_ids list to string
        """
        self.results_dir = directory
        self.csv_path = os.path.join(self.results_dir, csv_locator)
        self.doc_id_delimiter = doc_id_delimiter
        
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.base_columns = ["config_hash"]
        self.config_columns = [
            "model.primary",
            "model.allow_image_input",
            "graph_verification.enabled",
            "graph_verification.method",
            "graph_verification.llm_model",
            "prompt_mode.mode",
            "prompt_mode.elaborator_model",
            "text_summariser.model",
            "embedding.method",
            "embedding.model",
            "logger.level",
            "document_range.enabled",
            "document_range.document_ids",
            "rag.text_similarity_threshold",
            "iteration.loop_retries",
            "iteration.pass_threshold"
        ]
        self.all_columns = self.base_columns + self.config_columns + ["result_path"]
    
    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Create a hash of the config dictionary for unique identification."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:10]
    
    def check_hash_exists(self, config: Dict[str, Any]) -> tuple[bool, str]:
        """
        Check if a result already exists for the given config.
        
        :param config: The configuration dictionary to check
        :return: Tuple of (exists: bool, hash: str)
        """
        config_hash = self._hash_config(config)
        result_path = os.path.join(self.results_dir, f"{config_hash}.json")
        return os.path.exists(result_path), config_hash
    
    def _flatten_config(self, config: Dict[str, Any]) -> Dict[str, str]:
        """
        Flattens a nested configuration dictionary into a single-level dictionary.
        Special handling for document_ids which are converted to a delimited string.
        
        :param config: The nested configuration dictionary
        :return: Flattened configuration dictionary
        """
        flattened = {}
        
        def flatten(d, parent_key=""):
            for key, value in d.items():
                new_key = f"{parent_key}.{key}" if parent_key else key
                
                if key == "document_ids" and isinstance(value, list):
                    flattened[new_key] = self.doc_id_delimiter.join(map(str, value))
                elif isinstance(value, dict):
                    flatten(value, new_key)
                else:
                    flattened[new_key] = str(value)
        
        flatten(config)
        return flattened
    
    def write_result(self, config: dict, result: dict):
        """
        Writes results to a JSON file and updates the CSV tracker
        
        :param config: The configuration dictionary
        :param result: The result dictionary to be saved
        """
        config_hash = self._hash_config(config)
        
        result_path = os.path.join(self.results_dir, f"{config_hash}.json")
        
        with open(result_path, "w") as f:
            json.dump(result, indent=4, fp=f)
        
        flattened_config = self._flatten_config(config)
        
        self._update_csv_tracker(flattened_config, config_hash, result_path)
        
        return result_path
    
    def _update_csv_tracker(self, flattened_config: dict, config_hash: str, result_path: str):
        """
        Updates the CSV tracker with the new result information.
        Uses predefined columns from the known config structure with result_path at the end.
        No longer includes the full JSON config.
        
        :param flattened_config: Flattened configuration dictionary
        :param config_hash: Hash of the configuration
        :param result_path: Path to the saved result file
        """
        new_row_data = {
            "config_hash": config_hash,
            "result_path": result_path
        }
        
        for col in self.config_columns:
            new_row_data[col] = ""  
            
        for key, value in flattened_config.items():
            if key in self.config_columns:
                
                new_row_data[key] = value
        
        
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            
            for col in self.all_columns:
                if col not in df.columns:
                    df[col] = ""  
        else:
            df = pd.DataFrame(columns=self.all_columns)
        
        new_row = pd.DataFrame([new_row_data], columns=self.all_columns)
        

        df = pd.concat([df, new_row], ignore_index=True)
        
        df.to_csv(self.csv_path, index=False)