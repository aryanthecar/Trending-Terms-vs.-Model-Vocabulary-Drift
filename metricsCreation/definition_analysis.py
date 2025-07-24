import pandas as pd
import csv
from metrics import get_model_definition, compare_definitions
import google.generativeai as genai
import time
import os
import re
from typing import List, Dict
from tqdm import tqdm

GEMINI_API_KEY = 'INSERT GEMINI API KEY HERE'
INPUT_CSV = 'metricsCreation/combinedTerms.csv'
OUTPUT_CSV = 'definition_analysis_results.csv'
BATCH_SIZE = 25
LIMIT = None

MODELS = [
    'HuggingFaceTB/SmolLM2-135M', 
    "gpt2",
    "gemini-2.5-flash",
    "Qwen/Qwen3-0.6B", 
    'gemini-2.0-flash-lite',
    "google/gemma-3-1b-it", 
    "gemini-2.5-pro",
    'gemini-1.5-flash',
]

def get_first_sentence(text: str) -> str:
    """Extract the first sentence from text"""
    if not text:
        return text
    
    # Remove extra whitespace and newlines
    text = ' '.join(text.split())
    
    # Split on sentence-ending punctuation followed by space and capital letter
    sentences = re.split(r'[.!?]+\s+(?=[A-Z])', text)
    
    if sentences:
        first_sentence = sentences[0].strip()
        # Add period if it doesn't end with punctuation
        if first_sentence and not first_sentence[-1] in '.!?':
            first_sentence += '.'
        return first_sentence
    
    return text

def load_terms_data(csv_path: str, limit: int = None) -> List[Dict]:
    terms_data = []
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            terms_data.append({
                'word': row['word'],
                'reference_definition': row['standard_definition']
            })
    return terms_data

def batch_gemini_definitions(words: List[str], model_name: str, api_key: str) -> Dict[str, str]:
    if not api_key:
        return {word: None for word in words}
    
    all_definitions = {}
    
    for i in range(0, len(words), BATCH_SIZE):
        batch_words = words[i:i+BATCH_SIZE]
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
                
                batch_prompt = "Define these slang terms briefly:\n"
                for j, word in enumerate(batch_words, 1):
                    batch_prompt += f"{j}. {word}\n"
                
                response = model.generate_content(batch_prompt)
                
                if response.text:
                    lines = response.text.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if ':' in line and any(word.lower() in line.lower() for word in batch_words):
                            parts = line.split(':', 1)
                            if len(parts) == 2:
                                word_part = parts[0].strip()
                                def_part = parts[1].strip()
                                
                                for word in batch_words:
                                    if word.lower() in word_part.lower():
                                        # Trim to first sentence
                                        all_definitions[word] = get_first_sentence(def_part)
                                        break
                
                for word in batch_words:
                    if word not in all_definitions:
                        all_definitions[word] = None
                
                break 
                        
            except Exception as e:
                retry_count += 1
                if "quota" in str(e).lower() or "rate" in str(e).lower():
                    wait_time = 60 * retry_count  # Progressive wait: 60s, 120s, 180s
                    print(f"Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Batch error for {model_name}: {e}")
                    for word in batch_words:
                        all_definitions[word] = None
                    break
        
        # time delay between batches cuz i dont want to get rate limited
        time.sleep(10)  
    
    return all_definitions

def process_all_terms_and_models(terms_data: List[Dict], models: List[str], api_key: str = None) -> List[Dict]:
    results = []
    all_words = [term['word'] for term in terms_data]
    
    for model_name in models:
        print(f"Processing {model_name}")
        
        if "gemini" in model_name.lower():
            model_definitions = batch_gemini_definitions(all_words, model_name, api_key)
            
            for term_data in tqdm(terms_data, desc=model_name):
                word = term_data['word']
                model_def = model_definitions.get(word)
                
                similarity_score = None
                if model_def:
                    similarity_score = compare_definitions(model_def, term_data['reference_definition'])
                
                results.append({
                    'word': word,
                    'model': model_name,
                    'similarity_score': similarity_score
                })
        else:
            for term_data in tqdm(terms_data, desc=model_name):
                word = term_data['word']
                model_def = get_model_definition(model_name, word, api_key)
                
                # Trim to first sentence
                if model_def:
                    model_def = get_first_sentence(model_def)
                
                print(model_def)
                similarity_score = None
                if model_def:
                    similarity_score = compare_definitions(model_def, term_data['reference_definition'])
                
                results.append({
                    'word': word,
                    'model': model_name,
                    'similarity_score': similarity_score
                })
                
                time.sleep(0.1)
    
    return results

def save_results_to_csv(results: List[Dict], output_path: str):
    if not results:
        return
    
    df = pd.DataFrame(results)
    pivot_df = df.pivot_table(
        index='word', 
        columns='model', 
        values='similarity_score', 
        aggfunc='first'
    ).reset_index()
    
    pivot_df.columns.name = None
    pivot_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Results saved to {output_path}")

def main():
    terms_data = load_terms_data(INPUT_CSV, limit=LIMIT)
    if not terms_data:
        return
    
    print(f"Loaded {len(terms_data)} terms")
    
    results = process_all_terms_and_models(terms_data, MODELS, GEMINI_API_KEY)
    
    if not results:
        return
    
    save_results_to_csv(results, OUTPUT_CSV)
    
    df = pd.DataFrame(results)
    valid_scores = df[df['similarity_score'].notna()]
    
    if len(valid_scores) > 0:
        print(f"Average similarity: {valid_scores['similarity_score'].mean():.4f}")
        model_scores = valid_scores.groupby('model')['similarity_score'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        for model, (score, count) in model_scores.iterrows():
            print(f"{model}: {score:.4f} (n={count})")


main()