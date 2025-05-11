from src.vector_database import VectorService
class RAGStage:
    def __init__(self, vector_service: VectorService, k: int = 20):
        self.vector_service = vector_service
        self.k = k

    def get_rag_prompt(self, prompt: str) -> dict:
        results = self.vector_service.get_data(prompt, self.k)
        return [self.map_scores(result) for result in results]
    
    def get_rag_prompt_filtered(self, prompt: str, text_threshold: float = 0.5) -> dict:
        results = self.get_rag_prompt(prompt)
        return self.filter_results(results, text_threshold)
    
    def filter_results(self, results: list[dict], text_threshold: float = 0.5) -> list[dict]:
        # We keep everything in sorted order, if it's an image, we keep it regardless of score as they vary a lot
        # But for text, we only keep it if it's above the threshold
        filtered_results = []
        for result in results:
            if result["type"] == "image" or result["type"] == "attachment_image":
                filtered_results.append(result)
            elif result["type"] == "text":
                if result["score"] > text_threshold:
                    filtered_results.append(result)
        return filtered_results
    
    def map_scores(self, results: dict) -> dict:
        result =  {
            "vector_id": results["id"],
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
