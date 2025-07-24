from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import sys

# Add the metricsCreation directory to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'metricsCreation'))

from metrics import tokenization_drift, semantic_drift, definition_drift, count_tokens_openAI, count_tokens_transformers, count_tokens_gemini

app = Flask(__name__)
CORS(app)

# Sample trending terms for demonstration
TRENDING_TERMS = [
    "rizz", "girl dinner", "delulu", "beige flag", "situationship", 
    "sigma", "main character", "quiet luxury", "npc", "mid", "feral",
    "canon event", "ate and left no crumbs", "he's just a guy",
    "loud budgeting", "chronically online", "corecore", "ick",
    "male manipulator music", "Harvard travel ban", "girlboss"
]

# Available models for drift analysis
AVAILABLE_MODELS = {
    "OpenAI Models (via tiktoken)": [
        "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"
    ],
    "Hugging Face Models (via transformers)": [
        'HuggingFaceTB/SmolLM2-135M', "gpt2", "Qwen/Qwen3-0.6B", "google/gemma-3-1b-it"
    ],
    "Gemini Models": [
        "gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-2.5-pro", "gemini-1.5-flash"
    ]
}

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif obj is None:
        return None
    return obj

@app.route('/')
def index():
    """Serve the main dashboard page"""
    return render_template('index.html')

@app.route('/api/terms', methods=['GET'])
def get_terms():
    """Get all available trending terms"""
    return jsonify({
        "terms": TRENDING_TERMS,
        "count": len(TRENDING_TERMS)
    })

@app.route('/api/terms/<term>', methods=['GET'])
def get_term_info(term):
    """Get information about a specific term"""
    if term not in TRENDING_TERMS:
        return jsonify({"error": "Term not found"}), 404
    
    return jsonify({
        "term": term,
        "available": True
    })

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get all available models for drift analysis"""
    return jsonify(AVAILABLE_MODELS)

@app.route('/api/drift/tokenization', methods=['POST'])
def compute_tokenization_drift():
    """Compute tokenization drift between two models for a term"""
    data = request.json
    word = data.get("word")
    model1 = data.get("model1")
    model2 = data.get("model2")
    api_key = data.get("api_key")
    
    if not all([word, model1, model2]):
        return jsonify({"error": "Missing required parameters: word, model1, model2"}), 400
    
    try:
        score = tokenization_drift(model1, model2, word, geminiAPI=api_key)
        
        if score is None:
            return jsonify({"error": "Failed to compute tokenization drift"}), 500
        
        score = convert_numpy_types(score)
        
        return jsonify({
            "drift_type": "tokenization",
            "word": word,
            "model1": model1,
            "model2": model2,
            "score": score,
            "interpretation": "Score closer to 1 means more drift, closer to 0 means less drift"
        })
    except Exception as e:
        return jsonify({"error": f"Error computing drift: {str(e)}"}), 500

@app.route('/api/drift/semantic', methods=['POST'])
def compute_semantic_drift():
    """Compute semantic drift between two models for a term"""
    data = request.json
    word = data.get("word")
    model1 = data.get("model1")
    model2 = data.get("model2")
    api_key = data.get("api_key")
    
    if not all([word, model1, model2]):
        return jsonify({"error": "Missing required parameters: word, model1, model2"}), 400
    
    try:
        score = semantic_drift(model1, model2, word, geminiAPI=api_key)
        
        if score is None:
            return jsonify({"error": "Failed to compute semantic drift"}), 500
        
        score = convert_numpy_types(score)
        
        return jsonify({
            "drift_type": "semantic",
            "word": word,
            "model1": model1,
            "model2": model2,
            "score": score,
            "interpretation": "Score closer to 1 means more drift, closer to 0 means less drift"
        })
    except Exception as e:
        return jsonify({"error": f"Error computing drift: {str(e)}"}), 500

@app.route('/api/drift/definition', methods=['POST'])
def compute_definition_drift():
    """Compute definition drift between two models for a term"""
    data = request.json
    word = data.get("word")
    model1 = data.get("model1")
    model2 = data.get("model2")
    reference_definition = data.get("reference_definition", "")
    api_key = data.get("api_key")
    
    if not all([word, model1, model2]):
        return jsonify({"error": "Missing required parameters: word, model1, model2"}), 400
    
    if not reference_definition:
        reference_definition = f"Standard definition of the word '{word}'"
    
    try:
        score = definition_drift(model1, model2, word, reference_definition, geminiAPI=api_key)
        
        if score is None:
            return jsonify({"error": "Failed to compute definition drift"}), 500
        
        score = convert_numpy_types(score)
        
        return jsonify({
            "drift_type": "definition",
            "word": word,
            "model1": model1,
            "model2": model2,
            "score": score,
            "reference_definition": reference_definition,
            "interpretation": "Score closer to 1 means more drift, closer to 0 means less drift"
        })
    except Exception as e:
        return jsonify({"error": f"Error computing drift: {str(e)}"}), 500

@app.route('/api/tokens/<term>', methods=['GET'])
def get_token_count(term):
    """Get token count for a term using different models"""
    model = request.args.get('model', 'gpt2')
    api_key = request.args.get('api_key')
    
    # Trim whitespace from API key
    if api_key:
        api_key = api_key.strip()
    
    try:
        # OpenAI models that use tiktoken
        # i think gpt2 might be included in this list too
        tiktoken_models = {
            'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo', 'gpt-4o'
        }
        
        # Determine which tokenizer to use based on model
        if model in tiktoken_models:
            token_count = count_tokens_openAI(term, model)
        elif 'gemini' in model.lower():
            if not api_key:
                return jsonify({
                    "error": "Gemini API key is required for Gemini models"
                }), 400
         
            token_count = count_tokens_gemini(term, api_key, model)
        else:
            token_count = count_tokens_transformers(term, model)
        
        if token_count is None:
            return jsonify({
                "error": f"Failed to get token count for {model}"
            }), 500
        
        response = jsonify({
            "token_count": int(token_count) if token_count is not None else 0,
            "term": term,
            "model": model,
            "timestamp": pd.Timestamp.now().isoformat()
        })
        
        # Add no-cache headers
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        
        return response
        
    except Exception as e:
        return jsonify({
            "error": f"Error computing token count: {str(e)}"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Drift Analysis Dashboard API is running"
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 