# 📁 `research/` Directory  
## 🔍 Trending-Term Drift Explorer – Research Documentation

This folder contains all exploratory work, proof-of-concept notebooks, and documentation related 
to the research and experimentation phase of the **Trending-Term Drift Explorer** project. The 
purpose of this directory is to validate the feasibility of identifying linguistic drift in 
LLMs by researching methods for data scraping and extraction as well as evaluation. This directory
is solely research based and not representative of the final project. 

## 🧭 Research Goals as Completed

- Investigate and document potential API and tools to extract trending terms from Reddit and X (formerly Twitter)
- Define and explore potential method of LLM evaluation as  **token drift**, **semantic drift**, and **contextual drift**
- Identify and document tooling services that allow access to multiple LLMs (GPT-2, BERT, GPT-3.5-turbo, etc.)
--
## ✅ Completed Research & Files

### Data Extraction
#### Reddit
- For scraping and aquiring data from subreddits and Reddit posts, we found an API wrapper called [PRAW](https://praw.readthedocs.io/en/stable/getting_started/quick_start.html) that integrates with the [Reddit API](https://www.reddit.com/dev/api/)
- To be able to use these tools, a client_id and client_secret will need to be created in addition to a reddit account username and password
- Scraped data can then be exported text to '.txt' file for downstream token frequency analysis.

#### Twitter
- There are numerous tools for scraping data from Twitter
- The main API is Twitter's in-house API service called [X API V2](https://developer.x.com/en/docs/x-api)
- We also found two open source APIs that my allow easier access to Twitter Scraping:
  - [Tweepy](https://www.tweepy.org)
  - [snscrape](https://github.com/JustAnotherArchivist/snscrape)
 
#### Text Processing
- For text processing, if both scraped outputs from Twiiter and Reddit are in .txt files, a python script can be created to parse through the text, keep a count of all words, while eliminiation the top n words in the english dictionary, and the from there returning a list of the top X common words where X is the number of words wanted.

### Evaluation Methods

#### `tiktokenExploration.ipynb`
- Used OpenAI’s ['tiktoken'](https://github.com/openai/tiktoken) library to analyze token drift.
- Example: Compared how different slang terms (e.g., "liberalism", "rizz", "delulu") are tokenized.
- Result: Minimal token drift detected — led to de-emphasis of this method.

#### `semantic_drift_embeddings.ipynb`
- Loaded two Hugging Face OpenAI models ([GPT2](https://huggingface.co/docs/transformers/v4.18.0/model_doc/gpt2) and [OpenAI GPT](https://huggingface.co/docs/transformers/v4.18.0/model_doc/openai-gpt)
- Compared embeddings of slang term "chopped"
- Result: Identified drift score of 1 - cosineSimilarity as viable metric for semantic shift where lower scores indicate higher drift

#### `definition_similarity.ipynb`
- Prompted GPT-2 HF transformer for definition of chopped
- Compared outputs to Urban Dictionary definitions using:
  - [HFSentenceTransformers](https://huggingface.co/sentence-transformers)
  - ROUGE-L score via [rogue](https://pypi.org/project/rouge/)
- Example word tested: "chopped
- Found ability to score and rank variability based on intial result

