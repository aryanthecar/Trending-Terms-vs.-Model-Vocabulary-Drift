import pandas as pd
import csv
import numpy as np
from metrics import get_word_embedding, get_gemini_embedding
from sklearn.metrics.pairwise import euclidean_distances
import os
from typing import List, Dict
from tqdm import tqdm

GEMINI_API_KEY = 'INSERT GEMINI API KEY HERE'
INPUT_CSV = 'metricsCreation/combinedTerms.csv'
OUTPUT_CSV = 'embedding_drift_results.csv'
LIMIT = None

MODELS = [
    'HuggingFaceTB/SmolLM2-135M', 
    "gpt2",
    "gemini",
    "Qwen/Qwen3-0.6B", 
    "google/gemma-3-1b-it", 
]

def load_words(csv_path: str, limit: int = None) -> List[str]:
    words = []
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            words.append(row['word'])
    return words

def get_embedding(model_name: str, word: str, api_key: str) -> np.ndarray:
    if "gemini" in model_name.lower():
        return get_gemini_embedding(word, api_key)
    else:
        return get_word_embedding(model_name, word)

def align_vectors(vec1: np.ndarray, vec2: np.ndarray) -> tuple:
    if len(vec1) == len(vec2):
        return vec1, vec2
    
    max_len = max(len(vec1), len(vec2))
    
    if len(vec1) < max_len:
        vec1 = np.pad(vec1, (0, max_len - len(vec1)), mode='constant')
    
    if len(vec2) < max_len:
        vec2 = np.pad(vec2, (0, max_len - len(vec2)), mode='constant')
    
    return vec1, vec2

def calculate_drift(emb1: np.ndarray, emb2: np.ndarray) -> float:
    if emb1 is None or emb2 is None:
        return None
    
    emb1, emb2 = align_vectors(emb1, emb2)
    
    distance = euclidean_distances([emb1], [emb2])[0][0]
    
    max_possible_distance = np.linalg.norm(emb1) + np.linalg.norm(emb2)
    drift_score = distance / max_possible_distance
    
    return round(drift_score, 4)

def get_all_embeddings(words: List[str], models: List[str], api_key: str) -> Dict[str, Dict[str, np.ndarray]]:
    embeddings = {}
    
    for model_name in models:
        print(f"Getting embeddings for {model_name}")
        embeddings[model_name] = {}
        
        for word in tqdm(words, desc=model_name):
            emb = get_embedding(model_name, word, api_key)
            embeddings[model_name][word] = emb
    
    return embeddings

def calculate_all_drifts(words: List[str], models: List[str], embeddings: Dict) -> List[dict]:
    results = []
    
    for word in words:
        word_results = {'word': word}
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i < j:
                    emb1 = embeddings[model1][word]
                    emb2 = embeddings[model2][word]
                    
                    drift_score = calculate_drift(emb1, emb2)
                    pair_name = f"{model1}_vs_{model2}"
                    word_results[pair_name] = drift_score
        
        results.append(word_results)
    
    return results

def save_results(results: List[dict], output_path: str):
    if not results:
        return
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Results saved to {output_path}")

def main():
    words = load_words(INPUT_CSV, limit=LIMIT)
    if not words:
        return
    
    print(f"Loaded {len(words)} words")
    print(f"Using {len(MODELS)} models: {MODELS}")
    
    embeddings = get_all_embeddings(words, MODELS, GEMINI_API_KEY)
    
    results = calculate_all_drifts(words, MODELS, embeddings)
    
    if not results:
        return
    
    save_results(results, OUTPUT_CSV)
    
    df = pd.DataFrame(results)
    numeric_cols = [col for col in df.columns if col != 'word']
    
    for col in numeric_cols:
        valid_scores = df[col].dropna()
        if len(valid_scores) > 0:
            print(f"{col}: {valid_scores.mean():.4f}")

main()