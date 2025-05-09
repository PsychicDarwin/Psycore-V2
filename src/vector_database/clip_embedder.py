from transformers import CLIPProcessor, CLIPModel

class CLIPEmbedder:
    BASE_MODEL = "openai/clip-vit-base-patch32"
    PROCESSOR_MODEL = "openai/clip-vit-base-patch32"

    def __init__(self):