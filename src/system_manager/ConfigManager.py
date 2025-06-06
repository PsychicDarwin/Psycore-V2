import yaml
import os

class ConfigError(Exception):
    pass

class ConfigManager:
    VALID_GRAPH_METHODS = {"llm", "bert"}
    VALID_PROMPT_MODES = {"original", "elaborated", "q_learning","q_training"}
    VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    VALID_EMBEDDING_METHODS = {"langchain", "clip", "aws"}

    def __init__(self, path="config.yaml"):
        self.path = path
        self.config = self._load()
        self._validate()

    def _load(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Config file not found at: {self.path}")
        with open(self.path, "r") as f:
            return yaml.safe_load(f)

    def _validate(self):
        c = self.config

        
        for section in ["model", "graph_verification", "prompt_mode", "text_summariser", "logger", "embedding", "document_range", "rag", "iteration"]:
            if section not in c:
                raise ConfigError(f"Missing required section: '{section}'")

        
        if not isinstance(c["model"].get("primary"), str):
            raise ConfigError("model.primary must be a string")
        if not isinstance(c["model"].get("allow_image_input"), bool):
            raise ConfigError("model.allow_image_input must be a boolean")

        gv = c["graph_verification"]
        if not isinstance(gv.get("enabled"), bool):
            raise ConfigError("graph_verification.enabled must be a boolean")

        method = gv.get("method")
        if method not in self.VALID_GRAPH_METHODS:
            raise ConfigError(f"graph_verification.method must be one of {self.VALID_GRAPH_METHODS}")

        if method == "llm":
            if not isinstance(gv.get("llm_model"), str):
                raise ConfigError("graph_verification.llm_model must be a string when method is 'llm'")

        mode = c["prompt_mode"].get("mode")
        if mode not in self.VALID_PROMPT_MODES:
            raise ConfigError(f"prompt_mode.mode must be one of {self.VALID_PROMPT_MODES}")
        
        if not isinstance(c["prompt_mode"].get("elaborator_model"), str):
            raise ConfigError("prompt_mode.elaborator_model must be a string")

        if not isinstance(c["text_summariser"].get("model"), str):
            raise ConfigError("text_summariser.model must be a string")

        emb = c["embedding"]
        method = emb.get("method")
        if method not in self.VALID_EMBEDDING_METHODS:
            raise ConfigError(f"embedding.method must be one of {self.VALID_EMBEDDING_METHODS}")
        
        if method in ["langchain", "aws"]:
            if not isinstance(emb.get("model"), str):
                raise ConfigError(f"embedding.model must be a string when method is '{method}'")

        if not isinstance(c["logger"].get("level"), str):
            raise ConfigError("logger.level must be a string")
        if c["logger"]["level"] not in self.VALID_LOG_LEVELS:
            raise ConfigError(f"logger.level must be one of {self.VALID_LOG_LEVELS}")

        dr = c["document_range"]
        if not isinstance(dr.get("enabled"), bool):
            raise ConfigError("document_range.enabled must be a boolean")
        if not isinstance(dr.get("document_ids"), list):
            raise ConfigError("document_range.document_ids must be a list")
        if not all(isinstance(doc_id, int) for doc_id in dr["document_ids"]):
            raise ConfigError("document_range.document_ids must contain only integers")
        if not all(doc_id >= 0 for doc_id in dr["document_ids"]):
            raise ConfigError("document_range.document_ids must contain only non-negative integers")

        rag = c["rag"]
        if not isinstance(rag.get("text_similarity_threshold"), (int, float)):
            raise ConfigError("rag.text_similarity_threshold must be a number")
        if not 0 <= rag["text_similarity_threshold"] <= 1:
            raise ConfigError("rag.text_similarity_threshold must be between 0 and 1")

        iteration = c["iteration"]
        if not isinstance(iteration.get("loop_retries"), int):
            raise ConfigError("iteration.loop_retries must be an integer")
        if iteration["loop_retries"] < 0:
            raise ConfigError("iteration.loop_retries must be non-negative")
        
        if not isinstance(iteration.get("pass_threshold"), (int, float)):
            raise ConfigError("iteration.pass_threshold must be a number")
        if not 0 <= iteration["pass_threshold"] <= 1:
            raise ConfigError("iteration.pass_threshold must be between 0 and 1")

    def get_model(self):
        return self.config["model"]["primary"]

    def allow_images(self):
        return self.config["model"]["allow_image_input"]

    def is_graph_verification_enabled(self):
        return self.config["graph_verification"]["enabled"]

    def get_graph_method(self):
        return self.config["graph_verification"]["method"]

    def get_graph_llm_model(self):
        if self.get_graph_method() == "llm":
            return self.config["graph_verification"]["llm_model"]
        return None

    def get_prompt_mode(self):
        return self.config["prompt_mode"]["mode"]

    def get_elaborator_model(self):
        return self.config["prompt_mode"]["elaborator_model"]

    def get_text_summariser_model(self):
        return self.config["text_summariser"]["model"]

    def get_embedding_method(self):
        return self.config["embedding"]["method"]

    def get_embedding_model(self):
        method = self.get_embedding_method()
        if method in ["langchain", "aws"]:
            return self.config["embedding"]["model"]
        return None

    def get_log_level(self):
        return self.config["logger"]["level"]

    def is_document_range_enabled(self):
        return self.config["document_range"]["enabled"]

    def get_document_ids(self):
        """Get the list of document IDs to process."""
        return self.config["document_range"]["document_ids"]

    def get_rag_text_similarity_threshold(self):
        return self.config["rag"]["text_similarity_threshold"]

    def get_iteration_loop_retries(self):
        return self.config["iteration"]["loop_retries"]

    def get_iteration_pass_threshold(self):
        return self.config["iteration"]["pass_threshold"]

# Example usage
if __name__ == "__main__":
    try:
        config = ConfigManager("config.yaml")
        print("Primary Model:", config.get_model())
        print("Allow Images:", config.allow_images())
        print("Graph Method:", config.get_graph_method())
        print("Prompt Mode:", config.get_prompt_mode())
        print("Text Summariser Model:", config.get_text_summariser_model())
        print("Embedding Method:", config.get_embedding_method())
        print("Embedding Model:", config.get_embedding_model())
        print("Log Level:", config.get_log_level())
        print("Document Range Enabled:", config.is_document_range_enabled())
        print("Document Range:", config.get_document_ids())
        print("RAG Text Similarity Threshold:", config.get_rag_text_similarity_threshold())
        print("Iteration Loop Retries:", config.get_iteration_loop_retries())
        print("Iteration Pass Threshold:", config.get_iteration_pass_threshold())
    except (ConfigError, FileNotFoundError) as e:
        print(f"Config error:", e)
