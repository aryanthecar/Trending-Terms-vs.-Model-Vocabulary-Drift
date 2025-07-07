# Environment Setup Guide

## Overview
This project analyzes trending terms vs. model vocabulary drift using NLP and machine learning techniques. The environment is set up using Conda for better dependency management.

## Environment Details
- **Environment Name**: `trending-drift`
- **Python Version**: 3.11
- **Package Manager**: Conda + pip

## Quick Setup

### 1. Create and Activate Environment
```bash
# Create the environment
conda create -n trending-drift python=3.11 -y

# Activate the environment
conda activate trending-drift
```

### 2. Install Core Dependencies
```bash
# Install Jupyter and core packages via conda
conda install -c conda-forge jupyter notebook ipykernel -y

# Install ML and NLP packages via pip
pip install torch transformers sentence-transformers scikit-learn numpy pandas matplotlib seaborn

# Install data scraping and text processing packages
pip install praw tweepy snscrape tiktoken rouge nltk
```

### 3. Register Jupyter Kernel
```bash
# Register the environment as a Jupyter kernel
/Users/prabhav/anaconda3/envs/trending-drift/bin/python -m ipykernel install --user --name trending-drift --display-name "Trending Drift Research"
```

### 4. Alternative: Install from Requirements
```bash
# If you have the requirements.txt file
pip install -r requirements.txt
```

## Using the Environment

### Start Jupyter Notebook
```bash
conda activate trending-drift
jupyter notebook
```

### Start Jupyter Lab (Alternative)
```bash
conda activate trending-drift
jupyter lab
```

## Key Libraries Included

### Core ML/NLP
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformers library
- **Sentence-Transformers**: For semantic embeddings
- **Scikit-learn**: Machine learning utilities
- **Numpy/Pandas**: Data manipulation

### Data Scraping
- **PRAW**: Reddit API wrapper
- **Tweepy**: Twitter API wrapper
- **snscrape**: Alternative Twitter scraping

### Text Processing
- **tiktoken**: OpenAI's tokenizer
- **rouge**: Text similarity metrics
- **NLTK**: Natural language toolkit

### Visualization
- **Matplotlib**: Basic plotting
- **Seaborn**: Statistical visualizations

## Project Structure
```
Trending-Terms-vs.-Model-Vocabulary-Drift/
├── messageScraping/          # Data collection scripts
│   └── twitter.ipynb
├── researchAndValidity/      # Research notebooks
│   ├── definition_similarity.ipynb
│   ├── semantic_drift_embeddings.ipynb
│   └── tiktokenExploration.ipynb
├── requirements.txt          # Python dependencies
└── ENVIRONMENT_SETUP.md      # This file
```

## Troubleshooting

### Common Issues

1. **Kernel not found**: Make sure to register the kernel after installation
2. **Import errors**: Ensure you're using the correct conda environment
3. **GPU issues**: PyTorch is installed for CPU by default; install CUDA version if needed

### Environment Verification
```bash
# Check if environment is active
conda info --envs

# Check Python path
which python

# Test key imports
python -c "import torch; import transformers; print('Environment OK!')"
```

## Next Steps
1. Set up API keys for Reddit and Twitter (if needed)
2. Explore the research notebooks in `researchAndValidity/`
3. Start with the data collection scripts in `messageScraping/` 