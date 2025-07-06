# `metrics/` Directory  
## Trending-Term Drift Explorer – Metrics Implementation

This folder contains the core implementation files for calculating and analyzing linguistic drift 
across different Large Language Models (LLMs). These files provide comprehensive implementation for 
measuring **tokenization drift**, **semantic drift**, and **definition drift** between models through direct access, 
and api routes with support for OpenAI libaries (tiktoken), Hugging Face Transformers, and Google Gemini models.


## Completed Implementation Files

### `metrics.py` - Core Drift Calculation Engine
#### Tokenization Functions
- **`count_tokens_openAI()`** - Tokenizes text using OpenAI's tiktoken library for GPT models
- **`count_tokens_transformers()`** - Tokenizes text using Hugging Face AutoTokenizer for transformer models
- **`count_tokens_gemini()`** - Tokenizes text using Google Gemini API for Gemini models
- Supports popular models including GPT-2, GPT-4, BERT, RoBERTa, T5, BART, and Gemini variants (2.5-flash, 2.5-pro, 1.5)

#### Embedding Functions
- **`get_word_embedding()`** - Generates contextual embeddings from Hugging Face transformer models
- **`get_gemini_embedding()`** - Generates embeddings using Google Gemini Embeddings: "gemini-embedding-exp-03-07"
- Handles different model architectures from hugging face (encoder-only, decoder-only, encoder-decoder)
- Provides appropriate embedding extraction strategies for each model type

#### Definition Generation & Comparison
- **`get_model_definition()`** - Generates slang term definitions using text generation models
- **`compare_definitions()`** - Compares model definitions against reference definitions using:
  - Sentence transformer cosine similarity (all-MiniLM-L6-v2)
  - ROUGE-L score for lexical overlap
  - Returns averaged similarity score between the two scoring metrics

#### Drift Calculation Methods
- **`tokenization_drift()`** - Calculates drift based on token count differences between models
- **`semantic_drift()`** - Measures semantic drift using cosine similarity of embeddings
- **`definition_drift()`** - Compares how differently models define the same term
- All drift scores range from 0 (maximum drift) to 1 (identical)
- Two different models need to be passed in for drift to be calculated
- geminiAPI input is labeled null as default but can be passed in when using a gemini model

### `drift_api.py` - Flask 
#### REST API Endpoints
- **`POST /drift/tokenization`** - Calculate tokenization drift between two models
- **`POST /drift/semantic`** - Calculate semantic drift between two models
- **`POST /drift/definition`** - Calculate definition drift between two models
- The purpose of this is to serve as Flask backend implementation for our interactive dashboard where users can select models and input a word to caclulate drift between the models based on the word inputed. 

### `updateDataMetrics.py` - Batch Dataset Processing
#### Data Processing Functions
- **`add_metrics_columns()`** - Adds drift metrics columns to existing pandas DataFrame
  - Calculates GPT-2 and OpenAI token counts
  - Computes embedding similarity between GPT-2 and BERT models
  - Generates definition similarity scores against standard definitions created during data preprocessing phase

#### Column Additions
- **`gpt2_token`** - Token count using GPT-2 tokenizer
- **`open_gpt_token`** - Token count using OpenAI tokenizer (default: gpt-4o)
- **`embedding_similarity`** - Semantic drift score between GPT-2 and BERT embeddings
- **`gpt2_definition_similarity`** - Similarity between GPT-2 definition and standard definition
- **`opengpt_definition_similarity`** - Similarity between GPT-Neo definition and standard definition

