# `metricsCreation/` Directory  
## Trending-Term Drift Explorer – Metrics Implementation

This folder contains the comprehensive implementation for calculating and analyzing linguistic drift across different Large Language Models (LLMs). The codebase provides end-to-end functionality for measuring **tokenization drift**, **semantic drift**, and **definition drift** between models, along with data analysis scripts, visualization tools, and results files.

## Core Implementation Files

### `metrics.py` - Core Drift Calculation Engine
The foundational module containing all drift calculation functions and utilities.

#### Tokenization Functions
- **`count_tokens_openAI()`** - Tokenizes text using OpenAI's tiktoken library for GPT models
- **`count_tokens_transformers()`** - Tokenizes text using Hugging Face AutoTokenizer for transformer models
- **`count_tokens_gemini()`** - Tokenizes text using Google Gemini API for Gemini models
- Supports models: GPT-2, GPT-4 variants, BERT, RoBERTa, T5, BART, SmolLM, Qwen, Gemini variants

#### Embedding Functions
- **`get_word_embedding()`** - Generates contextual embeddings from Hugging Face transformer models
- **`get_gemini_embedding()`** - Generates embeddings using Google Gemini Embeddings API ("gemini-embedding-001")
- **`align_vectors()`** - Aligns embeddings of different dimensions using zero-padding
- Handles different model architectures (encoder-only, decoder-only, encoder-decoder)

#### Definition Generation & Comparison
- **`get_model_definition()`** - Generates slang term definitions using text generation models
- **`compare_definitions()`** - Compares model definitions against reference definitions using:
  - SentenceTransformers cosine similarity (all-MiniLM-L6-v2 model)
  - ROUGE-L score for lexical overlap
  - Formula: `(cosine_similarity + rouge_L_score) / 2`

#### Drift Calculation Methods
- **`tokenization_drift()`** - Calculates drift based on token count differences: `1 - |tokens1 - tokens2| / max(tokens1, tokens2)`
- **`semantic_drift()`** - Measures semantic drift using Euclidean distance of embeddings: `euclidean_distance(e₁, e₂) / (||e₁|| + ||e₂||)`
- **`definition_drift()`** - Compares definitional differences between models using reference definitions
- All drift scores normalized to [0,1] where 0 = identical, 1 = maximum drift

## Analysis Scripts

### `definition_analysis.py` - Model Definition Comparison
Comprehensive script for analyzing how different models define slang terms compared to reference definitions.

#### Features
- **Batch processing** - Handles multiple models and terms efficiently
- **Gemini batch optimization** - Uses batch API calls for cost-effective Gemini processing
- **Rate limiting** - Built-in delays and retry logic for API stability
- **Sentence trimming** - Extracts first sentence from definitions for consistency
- **Pivot table output** - Words as rows, models as columns with similarity scores

#### Models Supported
- HuggingFaceTB/SmolLM2-135M, GPT-2, Qwen/Qwen3-0.6B, Google/Gemma-3-1b-it
- Gemini models: 2.5-flash, 2.0-flash-lite, 2.5-pro, 1.5-flash

### `token_analysis.py` - Cross-Model Token Count Analysis
Analyzes tokenization differences across various model architectures.

#### Features
- **Multi-tokenizer support** - OpenAI tiktoken, HuggingFace transformers, Gemini API
- **Comprehensive model coverage** - 13 different models including GPT, Gemini, and open-source variants
- **Pivot table output** - Words as rows, models as columns with token counts

#### Models Analyzed
- OpenAI: GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-4, GPT-3.5-turbo, GPT-2
- HuggingFace: SmolLM2-135M, Qwen3-0.6B, Gemma-3-1b-it
- Gemini: 2.5-flash, 2.0-flash-lite, 2.5-pro, 1.5-flash

### `embedding_drift_analysis.py` - Semantic Drift Matrix
Calculates pairwise semantic drift between model embeddings for comprehensive drift analysis.

#### Features
- **Embedding caching** - Stores embeddings for efficient pairwise comparisons
- **Vector alignment** - Handles different embedding dimensions via zero-padding
- **Euclidean distance calculation** - Normalized by vector magnitudes
- **Comprehensive model coverage** - 5 core models with all pairwise combinations

## Visualization & Analysis Tools

### `visualizations.py` - Comprehensive Visualization Suite
Professional visualization script generating publication-quality graphs for analysis insights.

#### Generated Visualizations
1. **Definition Similarity Bar Graph** - Average similarity scores across models
2. **GPT Token Timeline** - Token counts across GPT models with release years (2019-2024)
3. **Token Count Split Graphs** - Two separate bar charts for model comparison
4. **Embedding Drift Matrices** - Individual graphs for each of 5 models vs others (teal blue theme)
5. **Gemini Analysis** - Combined tokenization and definition similarity trends
6. **Correlation Analysis** - Scatter plot of token count vs definition similarity
7. **Performance Heatmap** - Multi-metric model comparison matrix
8. **Definition Drift Graphs** - Absolute difference analysis for 5 models

## Usage Examples

### Running Analysis Scripts
```bash
# Generate definition similarity analysis
python definition_analysis.py

# Analyze token counts across models  
python token_analysis.py

# Calculate embedding drift matrices
python embedding_drift_analysis.py

# Generate all visualizations
python visualizations.py
```

### API Usage
```python
# Start Flask server
python drift_api.py

# Calculate drift via API
curl -X POST http://localhost:5000/drift/semantic \
  -H "Content-Type: application/json" \
  -d '{"word": "rizz", "model1": "gpt2", "model2": "bert-base-uncased"}'
```


