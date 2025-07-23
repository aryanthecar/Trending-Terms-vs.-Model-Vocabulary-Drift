import tiktoken
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
import google.generativeai as genai



"""
# List of popular models for drift analysis
    # Decoder-Only (GPT-style)
    "gpt2", 
    "EleutherAI/gpt-neo-1.3B", 
    "EleutherAI/gpt-j-6B",
    "mistralai/Mistral-7B-v0.1",  
    "huggyllama/llama-7b",
    "meta-llama/Llama-2-7b-hf", 

    # Encoder-Only (BERT-style)
    "bert-base-uncased", 
    "roberta-base", 
    "albert-base-v2",
    "distilbert-base-uncased",
    "xlm-roberta-base", 

    # Seq2Seq (T5/BART-style)
    "t5-small",
    "facebook/bart-base",

    # Lightweight / Multilingual / Specialized
    "camembert-base",       
    "bigscience/bloom-560m",
    "microsoft/codebert-base" 
"""

# Tokenization functions for different models
def count_tokens_openAI(word: str, tokenizer_model: str = 'gpt-4o') -> int:
    try:
        tokenizer = tiktoken.encoding_for_model(tokenizer_model)
        return len(tokenizer.encode(word))
    except Exception as e:
        print(f"Error loading tokenizer for {tokenizer_model}: {e}")
        return 0

def count_tokens_transformers(word: str, model_name: str = 'gpt2') -> int:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokens = tokenizer.tokenize(word)
        return len(tokens)
    except Exception as e:
        print(f"Error loading tokenizer for {model_name}: {e}")
        return 0
    
def count_tokens_gemini(word: str, geminiAPI: str, model_name: str = 'gemini-2.5-flash') -> int:
    try:
        genai.configure(api_key=geminiAPI)
        model = genai.GenerativeModel(model_name)
        response = model.count_tokens(word)
        return response.total_tokens
    except Exception as e:
        print(f"Error loading tokenizer for {model_name}: {e}")
        return 0


# Embedding function for different models from huggingface transformers
def get_gemini_embedding(word: str, geminiAPI: str) -> np.ndarray:
    try:
        genai.configure(api_key=geminiAPI)
        model = genai.GenerativeModel('gemini-embedding-exp-03-07')
        result = model.embed_content(word)
        return np.array(result.embedding)
    except Exception as e:
        print(f"Error generating embedding for {word} with Gemini: {e}")
        return None

def get_word_embedding(model_name: str, word: str) -> np.ndarray:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        # tokennize given input using the model's tokenizer
        inputs = tokenizer(word, return_tensors="pt")

        # Get model outputs
        with torch.no_grad():
            outputs = model(**inputs)

        # Decide which embedding to use based on model type
        if "t5" in model_name:
            # T5 is encoder-decoder; use mean of encoder output
            embedding = outputs.last_hidden_state.mean(dim=1)
        elif "gpt" in model_name:
            # use last token embedding for GPT-style models
            embedding = outputs.last_hidden_state[:, -1, :]
        else:
            # for encoder models (BERT, RoBERTa, etc.)
            if tokenizer.cls_token_id is not None:
                cls_index = (inputs.input_ids == tokenizer.cls_token_id).nonzero(as_tuple=True)[1]
                embedding = outputs.last_hidden_state[:, cls_index, :].squeeze(1)
            else:
                embedding = outputs.last_hidden_state.mean(dim=1)
        return embedding.squeeze().numpy()
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None


# Functions to get model definition using text generation and then compare it to a reference definition
def get_model_definition(model_name: str, word: str, geminiAPI: str = None) -> str:
    try:
        if "gemini" not in model_name.lower():
            pipe = pipeline("text-generation", model=model_name, tokenizer=model_name)
            prompt = f"Definition of the slang term '{word}':"
            
            result= pipe(
            prompt,
            max_new_tokens=30,
            do_sample=False,
            eos_token_id=pipe.tokenizer.eos_token_id,
            pad_token_id=pipe.tokenizer.pad_token_id
    )
            # result = pipe(prompt, max_new_tokens=50, do_sample=False, num_return_sequences=1)
            definition = result[0]["generated_text"]
        else:
            genai.configure(api_key=geminiAPI)
            model = genai.GenerativeModel(model_name)
            prompt = f"Definition of the slang term '{word}':"
            response = model.generate_content(prompt)
            definition = response.text
            prompt = f"Definition of the slang term '{word}':"
        
        # Clean up the definition by removing the prompt part
        if definition.startswith(prompt):
            definition = definition[len(prompt):].strip()
        return definition
    except Exception as e:
        print(f"{model_name} failed: {e}")
        return None
    

    
# Get similarity score based on model defintion vs. reference definition
def compare_definitions(model_def: str, reference_def: str) -> float:
    # find cosine similarity using sentence transformers model "all-MiniLM-L6-v2"
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb1 = model.encode(model_def, convert_to_tensor=True)
    emb2 = model.encode(reference_def, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item()
    transformerScore = round(score, 4)

    # find rouge score using rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference_def, model_def)
    rogueScore = round(scores["rougeL"].fmeasure, 4)

    # Return the average of both scores
    return (transformerScore + rogueScore) / 2
    


## COMPUTE DRIFT BETWEEN MODELS BASED ON DRIFT TYPE (TOKENIZATION, EMBEDDING, DEFINITION)

# THE DRIFT SCORE IS A VALUE BETWEEN 0 AND 1, WHERE 0 MEANS MAXIMUM DRIFT AND 1 MEANS NO DRIFT
# Essentially the drift score is the similarity between the two models for a given word, 
# where 1 means they are identical and 0 means they are completely different.
def tokenization_drift(model1_name: str, model2_name: str, word: str, geminiAPI: str = None) -> float:
    tiktokenModels = set([
        'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo', 
        'text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large'
    ])

    # tokens1 = counttoken transdofrmer if model1_name not in tiktokenModels else count_tokens_openAI(word, model1_name)
    tokens1 = count_tokens_openAI(word, model1_name) if model1_name in tiktokenModels else count_tokens_transformers(word, model1_name) if 'gemini' not in model1_name.lower() else count_tokens_gemini(word, geminiAPI, model1_name)
    tokens2 = count_tokens_openAI(word, model2_name) if model2_name in tiktokenModels else count_tokens_transformers(word, model2_name) if 'gemini' not in model2_name.lower() else count_tokens_gemini(word, geminiAPI, model2_name)

    if tokens1 == 0 or tokens2 == 0:
        print("Could not compute drift due to failed tokenization with given models.")
        return None

    drift_score = 1 - (abs(tokens1 - tokens2) / max(tokens1, tokens2))
    return round(drift_score, 4)


def semantic_drift(model1_name: str, model2_name: str, word: str, geminiAPI: str = None) -> float:
    # use gemini embedding if model is gemini, otherwise use huggingface transformers
    if "gemini" in model1_name.lower():
        emb1 = get_gemini_embedding(word, geminiAPI)
    else:
        emb1 = get_word_embedding(model1_name, word)

    if "gemini" in model2_name.lower():
        emb2 = get_gemini_embedding(word, geminiAPI)
    else:
        emb2 = get_word_embedding(model2_name, word)

    if emb1 is None or emb2 is None:
        print("Could not compute drift due to failed embedding with one or both of the given models.")
        return None

    sim = cosine_similarity([emb1],[emb2])[0][0]
    drift_score = 1 - sim
    drift_score = max(0.0, min(1.0, drift_score))  # Clamp to [0, 1]
    return round(drift_score, 4)


def definition_drift(model1_name: str, model2_name: str, word: str, reference_def: str, geminiAPI: str = None) -> float:
    def1 = get_model_definition(model1_name, word, geminiAPI)
    def2 = get_model_definition(model2_name, word, geminiAPI)

    if def1 is None or def2 is None:
        print("Could not compute drift due to failed definition generation with the provided models.")
        return None

    score1 = compare_definitions(def1, reference_def)
    score2 = compare_definitions(def2, reference_def)

    drift_score = 1 - abs(score1 - score2)
    return round(drift_score, 4)



modelDefinition = get_model_definition("Qwen/Qwen3-0.6B", "sybau")
print(modelDefinition)
    

