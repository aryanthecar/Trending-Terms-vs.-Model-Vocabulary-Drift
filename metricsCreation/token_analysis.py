import pandas as pd
import csv
from metrics import count_tokens_openAI, count_tokens_transformers, count_tokens_gemini
import time
import os
from typing import List, Dict
from tqdm import tqdm

GEMINI_API_KEY = 'INSERT GEMINI API KEY HERE'
INPUT_CSV = 'metricsCreation/combinedTerms.csv'
OUTPUT_CSV = 'token_analysis_results.csv'
LIMIT = None

MODELS = [
    'gpt-4o', 
    'gpt-4o-mini', 
    'gpt-4-turbo', 
    'gpt-4', 
    'gpt-3.5-turbo', 
    'gpt2',
    'HuggingFaceTB/SmolLM2-135M', 
    "gemini-2.5-flash",
    "Qwen/Qwen3-0.6B", 
    'gemini-2.0-flash-lite',
    "google/gemma-3-1b-it", 
    "gemini-2.5-pro",
    'gemini-1.5-flash',
]

OPENAI_MODELS = {'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo'}

def load_terms_data(csv_path: str, limit: int = None) -> List[str]:
    words = []
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            words.append(row['word'])
    return words

def get_token_count(word: str, model_name: str, api_key: str = None) -> int:
    if model_name in OPENAI_MODELS:
        return count_tokens_openAI(word, model_name)
    elif 'gemini' in model_name.lower():
        time.sleep(1)  # Rate limiting for Gemini
        return count_tokens_gemini(word, api_key, model_name)
    else:
        return count_tokens_transformers(word, model_name)

def process_all_terms_and_models(words: List[str], models: List[str], api_key: str = None) -> List[Dict]:
    results = []
    
    for model_name in models:
        print(f"Processing {model_name}")
        
        for word in tqdm(words, desc=model_name):
            token_count = get_token_count(word, model_name, api_key)
            
            results.append({
                'word': word,
                'model': model_name,
                'token_count': token_count
            })
    
    return results

def save_results_to_csv(results: List[Dict], output_path: str):
    if not results:
        return
    
    df = pd.DataFrame(results)
    pivot_df = df.pivot_table(
        index='word', 
        columns='model', 
        values='token_count', 
        aggfunc='first'
    ).reset_index()
    
    pivot_df.columns.name = None
    pivot_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Results saved to {output_path}")

def main():
    words = load_terms_data(INPUT_CSV, limit=LIMIT)
    print(f"Loaded {len(words)} words")
    results = process_all_terms_and_models(words, MODELS, GEMINI_API_KEY)
    if not results:
        return
    save_results_to_csv(results, OUTPUT_CSV)


main() 