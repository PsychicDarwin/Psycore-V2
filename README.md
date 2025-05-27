# Psycore

**Psycore** is the main engine that runs the full RAG pipeline — from setup to evaluation. It handles configuration loading, model selection, embedding setup, preprocessing, and prompt evaluation in one place.

It’s built for ablation studies, experimentation, and quick iteration on multimodal document pipelines.

---

## 🚀 What It Does

- Loads config (models, graphs, embeddings, etc.)
- Connects to S3 and Pinecone (or your vector DB)
- Preprocesses documents (image/textsummarisation, embedding, graph extraction)
- Runs RAG prompts (with elaboration and retrieval)
- Evaluates responses with BERTScore, ROUGE, and graph metrics
- Optionally retries to improve graph coverage

You can run it from Python, or use the CLI for fast testing.

---

## 🧠 Why Use Psycore?

- Fast setup via `config.yaml`
- Modular evaluation (choose your LLMs, embeddings, graph models)
- Clean separation between preprocessing and runtime
- Interactive prompt mode for manual testing
- Supports local and API-based workflows

---

## 🔧 Key Features

| Component        | Purpose |
|------------------|---------|
| `init_config()`  | Loads the model, embeddings, graph settings, and more |
| `init_s3()`      | Sets up the S3 buckets and credentials |
| `init_vector_database()` | Initializes the vector DB with the chosen embedding model |
| `preprocess()`   | Reconfigures the S3 buckets and vector DB, preprocesses documents for graph, image and text extraction/summarization |
| `process_prompt()` | Runs a prompt through RAG and outputs the answer + sources |
| `evaluate_prompt()` | Adds full evaluation (graph, BERTScore, ROUGE) with retry logic |
| `text_interface()` | Opens a CLI prompt loop for quick manual testing (basic implementation) |

---

## 🧪 Example Usage

```python
from psycore import Psycore

# Load the system from a config file
runner = Psycore(config_path="config.yaml")

# Preprocess documents (warning: wipes existing data)
runner.preprocess(skip_confirmation=True)

# Evaluate a prompt
result = runner.evaluate_prompt("What are the implications of the research?")
print(result["response"])

## 📦 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/PsychicDarwin/Psycore-V2
cd Psycore-V2
pip install -r requirements.txt
```
We recommend using a virtual environment to avoid conflicts with other projects.

# 🗂️ Project Structure — Psycore-V2

---

## Repository Structure
psycore-test-runner/
├── icons/ # Custom icons used in frontend or reports
├── jupyter_testing/ # Notebooks for testing and ablation runs
│ ├── config_variations/ # Different predefined test configs
│ │ ├── API_and_Hardware_Intensive/
│ │ ├── API_Limited/
│ │ └── General_Models/
│ └── results/ # Evaluation outputs from notebooks
├── src/ # Core implementation of the test runner
│ ├── data/ # File and data management
│ ├── evaluation/ # Evaluation methods (RAG, ROUGE, BERT, Graph)
│ ├── kg/ # Knowledge graph extraction and utilities
│ ├── llm/ # Language model interfaces and orchestration
│ ├── main/ # Entrypoint logic and high-level coordination
│ ├── preprocessing/ # Text/image preprocessing for summarization, embeddings
│ ├── results/ # Post-evaluation metrics and storage
│ ├── rl/ # QModel logic and reinforcement learning components
│ │ └── variations/ # Q-learning mode variants or experiments
│ ├── system_manager/ # Central config manager and control flow
│ └── vector_database/ # VDB interfaces (Pinecone, local, etc.)
├── uml_output/ # Auto-generated architecture diagrams
└── README.md # Main project overview and usage guide

# 📦 Primary class/program 'Psycore`

The `Psycore` class is the primary program and interface for the program. It uses all the models and has a directy versatile capability using config .yaml files to set up the system. It is designed to be modular and extensible, allowing for easy integration of new models and evaluation methods.


---

## 🔧 Initialization and Setup

When a `Psycore` instance is created, it performs the following:

- Loads configuration from a YAML file via `ConfigManager`
- Initialises model wrappers for:
  - Primary multimodal LLM
  - Text summarizer
  - Graph verification model
  - Prompt elaborator
- Authenticates and sets up:
  - Pinecone vector database
  - S3 buckets for document, image, graph, and text assets
- Loads any custom embedding models (Langchain, AWS Titan, or CLIP)

### Example

```python
from psycore import Psycore

runner = Psycore(config_path="config.yaml")
```
# Psycore-V2 PsycoreTestRunner Guide

## Critical Warnings

### Credentials & Resources
- For ablation studies, use throwaway credentials:
  - Pinecone index key
  - Graph bucket
  - Image bucket
  - Document text bucket
- **Rationale**: Prevents contamination of production data and allows for clean experimental results

### Model Limitations
- LocalModels guidelines:
  - Use models below 13B parameters
  - Not supported for GraphModel (JSON schema encoding issues)
  - Text summarization must use MLLM (required for image-to-text conversion)
- **Rationale**: Larger models are computationally expensive and may not be necessary for testing

### Processing Constraints
- Document processing limits:
  - Maximum 3 files for ablation studies
  - Each file takes ~20 minutes to preprocess
  - Cost: ~£0.30 in tokens per file
- **Rationale**: Balances testing efficiency with resource constraints

## 🚀 Quick Start

```python
from PsycoreTestRunner import PsycoreTestRunner

# Initialise with default config
runner = PsycoreTestRunner(preprocess=True)  # ⚠️ Preprocess will overwrite existing data
```

## ⚙️ Configuration

### Safe Configuration Modifications
The following configurations can be modified without requiring reprocessing:

1. **Runtime-Only Changes**
   - `prompt_mode.mode`: 
     - Switching between "elaborated" and "original"
     - Can use "q_learning" for testing with pre-trained models
     - "q_training" should only be used in production systems
     - Note: Using a pre-trained Q-learning model with a different elaborator model than what is was trained on may produce different results due to model-specific elaboration styles
   - `prompt_mode.elaborator_model`: Changing the elaborator model
   - `rag.text_similarity_threshold`: Adjusting the similarity threshold for filtering
   - `document_range`: Modifying the range of documents to process

2. **Changes Requiring Reprocessing**
   - `embedding.method` and `embedding.model`: Changes vector database format
   - `graph_verification.method` and `graph_verification.llm_model`: Affects graph structure
   - `text_summariser.model`: Only affects preprocessing, runtime changes have no effect

3. **Partial Impact Changes**
   - `text_summariser.model`: 
     - Can be changed at runtime without breaking
     - Changes only affect new preprocessing
     - Existing summaries remain unchanged
   - `document_range`:
     - Can be modified without reprocessing
     - Only affects which documents are processed
     - No impact on existing processed documents
   - Q-learning related settings:
     - Can be changed freely at runtime
     - May have learned preferences for specific model elaboration styles
     - Training data might be biased towards particular model outputs
     - No impact on preprocessed data or system stability
     - Pre-trained models may perform differently with different elaborator models

### Model Selection Guide

#### Available Models
Models are categorized in [ModelCatalogue](src/llm/model_catalogue.py):

1. **Multimodal Models (MLLMs)**
   - OpenAI: `oai_4o_latest`, `oai_chatgpt_latest`
   - Bedrock: `claude_3_sonnet`, `claude_3_haiku`
   - Gemini: `gemini_1.5_flash`, `gemini_1.5_8b_flash`, `gemini_1.5_pro`
   - Local: `llava_7b`, `llava_13b`, `llava_34b`, `bakllava_7b`

2. **Text Models**
   - OpenAI: `oai_3.5_final`
   - Bedrock: `meta_llama_3_70b_instruct`, `meta_llama_3_8b_instruct`
   - Local: Various DeepSeek, Qwen, and Phi models

3. **Embedding Models**
   - AWS: `bedrock_multimodal_g1_titan` (recommended for multi-modal)
   - OpenAI: `oai_text_3_large`
   - Local: `bge_m3`

#### JSON Schema Support
For graph verification, use models with JSON schema support:
- `oai_4o_latest`
- `oai_3.5_final`
- `claude_3_sonnet`
- `claude_3_haiku`
- `grok_2_text`
- `deepseek_70b_r1`
- `qwen_3b_2.5`

### Complete Configuration Example
```python
config = {
    "model": {
        "primary": "oai_4o_latest",     # Primary model
        "allow_image_input": True       # Enable image inputs
    },
    "graph_verification": {
        "enabled": True,
        "method": "bert",              # Options: "bert" or "llm"
        "llm_model": "oai_4o_latest"   # Required for "llm" method
    },
    "prompt_mode": {
        "mode": "elaborated",          # Options: "elaborated" or "original"
        "elaborator_model": "oai_4o_latest"
    },
    "text_summariser": {
        "model": "llava_13b"           # Text summarization model
    },
    "embedding": {
        "method": "aws",              # Options: "aws" or "clip"
        "model": "amazon.titan-embed-image-v1"  # Required for AWS
    },
    "document_range": {
        "enabled": True,
        "start_index": 0,
        "end_index": 3                # Recommended max: 3 files
    },
    "rag": {
        "text_similarity_threshold": 0.55  # Threshold for RAG retrieval
    }
}

# Apply configuration
runner.update_config(config, preprocess=True)
```

## 📊 Evaluation Methods

### Running Tests

#### Single Prompt Evaluation
```python
# Basic evaluation
result = runner.evaluate_prompt("Your test prompt")

# Access evaluation metrics
print(f"RAG Score: {result['score']}")
print(f"Graph Verification: {result['graph_verification']}")
print(f"BERTScore: {result['bert_score']}")
print(f"ROUGE Score: {result['rouge_score']}")
```

#### Multiple Prompts Evaluation
```python
# Batch evaluation
prompts = [
    "What are the key findings in the document?",
    "Summarize the main arguments presented.",
    "What are the implications of these findings?"
]
results = runner.evaluate_prompts(prompts)

# Process results
for i, result in enumerate(results):
    print(f"\nResults for prompt {i+1}:")
    print(f"RAG Score: {result['score']}")
    print(f"Graph Verification: {result['graph_verification']}")
    print(f"BERTScore: {result['bert_score']}")
    print(f"ROUGE Score: {result['rouge_score']}")
```

### Evaluation Metrics Explained

1. **RAG Retrieval**
   - Measures relevance of retrieved documents
   - Uses text similarity threshold (default: 0.55)
   - Returns top 10 most relevant documents

2. **Graph Verification**
   - BERT method: Uses REBEL model for relation extraction
   - LLM method: Uses JSON schema-capable models
   - Evaluates semantic relationships in text

3. **BERTScore**
   - Semantic similarity evaluation
   - Uses BERT embeddings for comparison
   - Provides F1, precision, and recall scores

4. **ROUGE Score**
   - Text overlap evaluation
   - Uses ROUGE-L metric
   - Measures n-gram overlap between texts

## 🎯 Technical Design Decisions & Best Practices

### Model Selection Rationale

#### Primary Model
- Any Multimodal Model from [ModelCatalogue](src/llm/model_catalogue.py)
- **Rationale**: Ensures comprehensive understanding of both text and visual content

#### Graph Verification
- LLM: Models with JSON schema support
- BERT: Local model (requires GPU)
- **Rationale**: BERT preferred for production due to token efficiency and speed

#### Text Summarization
- LLAVA7B recommended
- **Rationale**: Local execution reduces costs while maintaining quality

#### Embedding Methods
- AWS: `amazon.titan-embed-image-v1`
  - **Pros**: Better multi-modal retrieval
  - **Cons**: Higher cost due to external API
- CLIP: Local execution
  - **Pros**: Cost-effective, local processing
  - **Cons**: Less optimal multi-modal unification

### Performance Optimization Guidelines

1. **Text Summarization**
   - Use LLAVA7B for cost-effective processing
   - **Rationale**: Reduces token consumption while maintaining quality

2. **Graph Extraction**
   - Prefer BERT for production use
   - **Rationale**: More efficient than LLM-based extraction for large datasets

3. **Embedding Selection**
   - AWS vs CLIP decision based on:
     - Retrieval quality requirements
     - Cost constraints
     - Processing time requirements

## 🔄 Complete Workflow Example

### Base System Process
The system follows this workflow when processing prompts:

1. **Prompt Processing**
   ```python
   # 1. Elaboration Stage
   elaborator = RAGElaborator(self.elaborator_model)
   elaborated_prompt = elaborator.elaborate(base_prompt)
   
   # 2. Prompt Selection
   prompt_stage = PromptStage(None, self.prompt_style)
   chosen_prompt, elaborated = prompt_stage.decide_between_prompts(
       base_prompt, 
       elaborated_prompt
   )
   ```

2. **RAG Retrieval**
   ```python
   # 3. RAG Stage
   rag_stage = RAGStage(self.vdb, 10)  # Top 10 results
   rag_results = rag_stage.get_rag_prompt_filtered(
       chosen_prompt, 
       self.rag_text_similarity_threshold
   )
   ```

3. **Response Generation**
   ```python
   # 4. Chat Stage
   rag_chat_results = self.rag_chat.chat(base_prompt, rag_results)
   ```

4. **Evaluation (if using evaluate_prompt)**
   ```python
   # 5. Evaluation Stage
   evaluators = [
       GraphEvaluator(self.graph_model, self.s3_quick_fetch),
       BERTEvaluator(self.s3_quick_fetch),
       RougeEvaluator(self.s3_quick_fetch)
   ]
   for result in rag_results:
       for evaluator in evaluators:
           result = evaluator.evaluate_rag_result(
               rag_chat_results.content, 
               result
           )
   ```

### Process Flow Details

1. **Elaboration Stage**
   - Takes the base prompt
   - Uses the elaborator model to enhance the prompt
   - Maintains a history of previous prompts (max 6 by default)
   - Returns an elaborated version of the prompt

2. **Prompt Selection Stage**
   - Decides between original and elaborated prompts
   - Modes:
     - `original`: Uses base prompt
     - `elaborated`: Uses elaborated prompt
     - `q_learning`: Uses pre-trained model (not implemented)
     - `q_training`: Trains model (production only)
   - These prompts decide if the system should elaborate to improve RAG retrieval, to help issues like assumed context/tokens where 2018 to 2023 only has tokens 2018 and 2023 not 2019, 2020, 2021, 2022
   - The Q-learning model compares the original and elaborated prompts and decides which one to use based on the best RAG retrieval score
3. **RAG Stage**
   - Queries vector database with chosen prompt
   - Filters results based on similarity threshold
   - Returns top 10 or less results above threshold (images aren't evalauted against threshold and all in top 10 vectors are returned)
   - Each result includes:
     - Document path
     - Vector ID
     - Similarity score

4. **Chat Stage**
   - Takes base prompt and RAG results
   - Generates response using primary model
   - Chat history is also preserved in the [ChatAgent](src/llm/chat_agent.py) with the [ChatHistory](src/llm/chat_history.py) object
   - Returns formatted response with sources

5. **Evaluation Stage** (evaluate_prompt only)
   - Graph verification using selected method
   - BERTScore semantic similarity
   - ROUGE score for text overlap
   - Returns comprehensive evaluation metrics

### Example Usage
```python
# Initialie with custom config
config = {
    "model": {"primary": "oai_4o_latest"},
    "document_range": {"start_index": 0, "end_index": 3}
}
runner = PsycoreTestRunner(config=config, preprocess=True)

# Run evaluation
prompts = [
    "What are the key findings in the document?",
    "Summarize the main arguments presented."
]
results = runner.evaluate_prompts(prompts)

# Process results
for result in results:
    print(f"RAG Score: {result['score']}")
    print(f"Graph Verification: {result['graph_verification']}")
    print(f"BERTScore: {result['bert_score']}")
    print(f"ROUGE Score: {result['rouge_score']}")
```

## 🚧 Implementation Status

### Current Limitations
- QModel not in use (pending full implementation/integration)
- Use either elaborated mode or default mode
- `process_evaluation()` function pending implementation
  - Will consolidate scores across files when available

### Preprocessing Considerations
- ⚠️ Preprocessing overwrites existing data
- Ensure correct configuration before preprocessing
- VDB and buckets will be reset
- Preprocessed files may take hours to restore

## 📝 Research Notes

### Current Architecture Decisions
1. **Text Summarization**
   - LLAVA7B as primary choice
   - Local execution for cost reduction
   - MLLM requirement for image-to-text conversion

2. **Graph Extraction**
   - LLM ideal but impractical for full dataset
   - BERT preferred for production use
   - Token efficiency considerations

3. **Embedding Methods**
   - CLIP: Local but suboptimal unification
   - AWS: Better retrieval but higher cost
   - Ongoing ablation studies recommended

### Future Considerations
- Summary_path field needed for Rouge/BERTScore evaluation
- Ongoing evaluation of embedding methods
- Potential for QModel integration
- Optimization of preprocessing pipeline
