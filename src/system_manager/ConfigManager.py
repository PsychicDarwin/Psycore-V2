import yaml
import os

class ConfigError(Exception):
    pass

class ConfigManager:
    VALID_GRAPH_METHODS = {"llm", "bert"}
    VALID_PROMPT_MODES = {"original", "elaborated", "q_learning"}

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

        # Check presence
        for section in ["model", "graph_verification", "prompt_mode"]:
            if section not in c:
                raise ConfigError(f"Missing required section: '{section}'")

        # Validate model
        if not isinstance(c["model"].get("primary"), str):
            raise ConfigError("model.primary must be a string")
        if not isinstance(c["model"].get("allow_image_input"), bool):
            raise ConfigError("model.allow_image_input must be a boolean")

        # Validate graph_verification
        gv = c["graph_verification"]
        if not isinstance(gv.get("enabled"), bool):
            raise ConfigError("graph_verification.enabled must be a boolean")

        method = gv.get("method")
        if method not in self.VALID_GRAPH_METHODS:
            raise ConfigError(f"graph_verification.method must be one of {self.VALID_GRAPH_METHODS}")

        if method == "llm":
            if not isinstance(gv.get("llm_model"), str):
                raise ConfigError("graph_verification.llm_model must be a string when method is 'llm'")

        # Validate prompt_mode
        mode = c["prompt_mode"].get("mode")
        if mode not in self.VALID_PROMPT_MODES:
            raise ConfigError(f"prompt_mode.mode must be one of {self.VALID_PROMPT_MODES}")

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

# Example usage
if __name__ == "__main__":
    try:
        config = ConfigManager("config.yaml")
        print("Primary Model:", config.get_model())
        print("Allow Images:", config.allow_images())
        print("Graph Method:", config.get_graph_method())
        print("Prompt Mode:", config.get_prompt_mode())
    except (ConfigError, FileNotFoundError) as e:
        print(f"Config error:", e)
