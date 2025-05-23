import pandas as pd
import json

class ResultVisualiser:
    def __init__(self, results_path):
        self.results_path = results_path
        # Results csv
        self.results_csv = pd.read_csv(results_path+"/results.csv")
        self.configs = []
        for index, row in self.results_csv.iterrows():
            config = self.read_line(row)
            result_path = row["result_path"]
            with open(result_path, "r") as f:
                result = json.load(f)
            weighted_results = self.weight_results(result)
            config['results'] = weighted_results
            self.configs.append(config)

    def read_line(self, row):
        results_dict = {
            "model": {
                "primary": row["model.primary"],
                "allow_image_input": row["model.allow_image_input"],
            },
            "graph_verification": {
                "enabled": row["graph_verification.enabled"],
                "method": row["graph_verification.method"],
                "llm_model": row["graph_verification.llm_model"]
            },
            "prompt_mode": {
                "mode": row["prompt_mode.mode"],
                "elaborator_model": row["prompt_mode.elaborator_model"]
            },
            "text_summariser": {
                "model": row["text_summariser.model"]
            },
            "embedding": {
                "method": row["embedding.method"],
                "model": row["embedding.model"]
            },
            "rag": {
                "text_similarity_threshold": row["rag.text_similarity_threshold"],
                "loop_retries": row["iteration.loop_retries"],
                "pass_threshold": row["iteration.pass_threshold"]
            },
            "iteration": {
                "loop_retries": row["iteration.loop_retries"],
                "pass_threshold": row["iteration.pass_threshold"]
            }
        }
        return results_dict            

    def weight_results(self, result_dict):
        prompt_index = {}
        for prompt in result_dict:
            prompt_index[prompt] = {}
            response = result_dict[prompt]["response"]
            retry_count = result_dict[prompt]["retry_count"]
            chosen_prompt = result_dict[prompt]["chosen_prompt"]
            rag_results = result_dict[prompt]["rag_results"]
            rag_count = len(rag_results)
            # Sort all rag results by score
            rag_results.sort(key=lambda x: x["score"], reverse=False)
            final_results = {
                "retry_count": retry_count,
                "rag_count": rag_count,
                "graph_evaluation": {
                    "recall": 0,
                    "precision": 0,
                    "f_beta": 0,
                    "beta": 1.0
                },
                "bertscore_evaluation": {
                    "precision": 0,
                    "recall": 0,
                    "f1": 0
                },
                "rouge_evaluation": {
                    "rougeL": [0, 0, 0]
                }
            }
            for i, rag_result in enumerate(rag_results):
                # We multiply the score by the index + 1
                final_results["graph_evaluation"]["recall"] += rag_result["graph_evaluation"]["recall"] * (i + 1)
                final_results["graph_evaluation"]["precision"] += rag_result["graph_evaluation"]["precision"] * (i + 1)
                final_results["graph_evaluation"]["f_beta"] += rag_result["graph_evaluation"]["f_beta"] * (i + 1)
                final_results["bertscore_evaluation"]["precision"] += rag_result["bertscore_evaluation"]["precision"][0] * (i + 1)
                final_results["bertscore_evaluation"]["recall"] += rag_result["bertscore_evaluation"]["recall"][0] * (i + 1)
                final_results["bertscore_evaluation"]["f1"] += rag_result["bertscore_evaluation"]["f1"][0] * (i + 1)
                for j in range(len(final_results["rouge_evaluation"]["rougeL"])):
                    final_results["rouge_evaluation"]["rougeL"][j] += rag_result["rouge_evaluation"]["rougeL"][j] * (i + 1)

            final_results["graph_evaluation"]["recall"] /= (rag_count * (rag_count + 1) / 2)
            final_results["graph_evaluation"]["precision"] /= (rag_count * (rag_count + 1) / 2)
            final_results["graph_evaluation"]["f_beta"] /= (rag_count * (rag_count + 1) / 2)
            final_results["bertscore_evaluation"]["precision"] /= (rag_count * (rag_count + 1) / 2)
            final_results["bertscore_evaluation"]["recall"] /= (rag_count * (rag_count + 1) / 2)
            final_results["bertscore_evaluation"]["f1"] /= (rag_count * (rag_count + 1) / 2)
            for j in range(len(final_results["rouge_evaluation"]["rougeL"])):
                final_results["rouge_evaluation"]["rougeL"][j] /= (rag_count * (rag_count + 1) / 2)

            prompt_index[prompt] = final_results
        result = {
            "response": response,
            "chosen_prompt": chosen_prompt,
            "rag_count": rag_count,
            "final_results": final_results
        }
        return result


if __name__ == "__main__":
    visualiser = ResultVisualiser("results")
    print(visualiser.configs[0])