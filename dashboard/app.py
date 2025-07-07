from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import os
import sys

# Add the metricsCreation directory to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'metricsCreation'))

from metrics import tokenization_drift, semantic_drift, definition_drift, count_tokens_openAI, count_tokens_transformers

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

# Available models for drift analysis - using more compatible models
AVAILABLE_MODELS = {
    "OpenAI Models (via tiktoken)": [
        "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"
    ],
    "Hugging Face Models (via transformers)": [
        "gpt2", "bert-base-uncased", "roberta-base", "distilbert-base-uncased",
        "t5-small", "facebook/bart-base", "albert-base-v2"
    ]
}

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
    
    if not all([word, model1, model2]):
        return jsonify({"error": "Missing required parameters: word, model1, model2"}), 400
    
    try:
        # For now, we'll use None for gemini_api since we're not implementing API keys yet
        score = tokenization_drift(model1, model2, word, geminiAPI=None)
        
        if score is None:
            return jsonify({"error": "Failed to compute tokenization drift"}), 500
        
        return jsonify({
            "drift_type": "tokenization",
            "word": word,
            "model1": model1,
            "model2": model2,
            "score": score,
            "interpretation": "Score closer to 1 means less drift, closer to 0 means more drift"
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
    
    if not all([word, model1, model2]):
        return jsonify({"error": "Missing required parameters: word, model1, model2"}), 400
    
    try:
        # For now, we'll use None for gemini_api since we're not implementing API keys yet
        score = semantic_drift(model1, model2, word, geminiAPI=None)
        
        if score is None:
            return jsonify({"error": "Failed to compute semantic drift"}), 500
        
        return jsonify({
            "drift_type": "semantic",
            "word": word,
            "model1": model1,
            "model2": model2,
            "score": score,
            "interpretation": "Score closer to 1 means less drift, closer to 0 means more drift"
        })
    except Exception as e:
        return jsonify({"error": f"Error computing drift: {str(e)}"}), 500

@app.route('/api/tokens/<term>', methods=['GET'])
def get_token_count(term):
    """Get token count for a term using different models"""
    model = request.args.get('model', 'gpt2')
    
    try:
        # Determine which tokenizer to use based on model
        tiktoken_models = {
            'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo'
        }
        
        if model in tiktoken_models:
            token_count = count_tokens_openAI(term, model)
        else:
            token_count = count_tokens_transformers(term, model)
        
        return jsonify({
            "term": term,
            "model": model,
            "token_count": token_count
        })
    except Exception as e:
        return jsonify({"error": f"Error counting tokens: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Drift Analysis Dashboard API is running"
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 