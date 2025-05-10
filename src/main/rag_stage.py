from src.vector_database import VectorService
class RAGStage:
    def __init__(self, vector_service: VectorService, k: int = 10):
        self.vector_service = vector_service
        self.k = k

    def get_rag_prompt(self, prompt: str) -> dict:
        results = self.vector_service.get_data(prompt, self.k)
        return [self.map_scores(result) for result in results]
    
    def map_scores(self, results: dict) -> dict:
        result =  {
            "score": results["score"],
            "document_path": results["metadata"]["document_path"],
            "graph_path": results["metadata"]["graph_path"],
            "type": results["metadata"]["type"]
        }
        if result["type"] == "text":
            result["text"] = results["metadata"]["text"]
        elif result["type"] == "image":
            result["image_path"] = results["metadata"]["image_path"]
        elif result["type"] == "attachment_image":
            result["image_path"] = results["metadata"]["image_path"]
        return result
