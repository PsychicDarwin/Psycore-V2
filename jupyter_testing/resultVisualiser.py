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
                },
                "chosen_prompt": chosen_prompt,
                "response": response,
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
        return prompt_index

    def save_results_to_csv(self, output_path):
        """Save the weighted results to a CSV file."""
        rows = []
        for config in self.configs:
            for prompt, results in config['results'].items():
                # Clean and escape text for CSV
                original_prompt = str(prompt).replace('\n', ' ').replace('\r', ' ').strip()
                chosen_prompt = str(results['chosen_prompt']).replace('\n', ' ').replace('\r', ' ').strip()
                response = str(results['response']).replace('\n', ' ').replace('\r', ' ').strip()
                
                row = {
                    'original_prompt': original_prompt,
                    'chosen_prompt': chosen_prompt,
                    'response': response,
                    'model_primary': config['model']['primary'],
                    'model_allow_image_input': config['model']['allow_image_input'],
                    'graph_verification_enabled': config['graph_verification']['enabled'],
                    'graph_verification_method': config['graph_verification']['method'],
                    'graph_verification_llm_model': config['graph_verification']['llm_model'],
                    'prompt_mode': config['prompt_mode']['mode'],
                    'prompt_mode_elaborator': config['prompt_mode']['elaborator_model'],
                    'text_summariser_model': config['text_summariser']['model'],
                    'embedding_method': config['embedding']['method'],
                    'embedding_model': config['embedding']['model'],
                    'rag_text_similarity_threshold': config['rag']['text_similarity_threshold'],
                    'rag_loop_retries': config['rag']['loop_retries'],
                    'rag_pass_threshold': config['rag']['pass_threshold'],
                    'retry_count': results['retry_count'],
                    'rag_count': results['rag_count'],
                    'graph_recall': results['graph_evaluation']['recall'],
                    'graph_precision': results['graph_evaluation']['precision'],
                    'graph_f_beta': results['graph_evaluation']['f_beta'],
                    'bertscore_precision': results['bertscore_evaluation']['precision'],
                    'bertscore_recall': results['bertscore_evaluation']['recall'],
                    'bertscore_f1': results['bertscore_evaluation']['f1'],
                    'rougeL_precision': results['rouge_evaluation']['rougeL'][0],
                    'rougeL_recall': results['rouge_evaluation']['rougeL'][1],
                    'rougeL_f1': results['rouge_evaluation']['rougeL'][2]
                }
                rows.append(row)
                print(f"Added row for original prompt: {original_prompt[:100]}...")  # Print first 100 chars of prompt
        
        df = pd.DataFrame(rows)
        # Ensure proper CSV escaping
        df.to_csv(output_path, index=False, quoting=1)  # QUOTE_ALL mode
        print(f"\nResults saved to {output_path}")
        print(f"Total rows saved: {len(rows)}")
        print("\nFirst few rows of data:")
        print(df.head())


if __name__ == "__main__":
    visualiser = ResultVisualiser("results")
    visualiser.save_results_to_csv("results/weighted_results.csv")