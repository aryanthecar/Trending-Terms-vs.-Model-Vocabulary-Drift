from flask import Flask, request, jsonify
from metrics import tokenization_drift, semantic_drift, definition_drift

app = Flask(__name__)

@app.route('/drift/tokenization', methods=['POST'])
def drift_tokenization():
    data = request.json
    word = data.get("word")
    model1 = data.get("model1")
    model2 = data.get("model2")
    gemini_key = data.get("gemini_api")

    if not all([word, model1, model2]):
        return jsonify({"error": "Missing required parameters"}), 400

    score = tokenization_drift(model1, model2, word, gemini_key)
    return jsonify({"drift_type": "tokenization", "score": score})

@app.route('/drift/semantic', methods=['POST'])
def drift_semantic():
    data = request.json
    word = data.get("word")
    model1 = data.get("model1")
    model2 = data.get("model2")
    gemini_key = data.get("gemini_api")

    if not all([word, model1, model2]):
        return jsonify({"error": "Missing required parameters"}), 400

    score = semantic_drift(model1, model2, word, gemini_key)
    return jsonify({"drift_type": "semantic", "score": score})

@app.route('/drift/definition', methods=['POST'])
def drift_definition():
    data = request.json
    word = data.get("word")
    model1 = data.get("model1")
    model2 = data.get("model2")
    reference_def = data.get("reference_def")
    gemini_key = data.get("gemini_api")

    if not all([word, model1, model2, reference_def]):
        return jsonify({"error": "Missing required parameters"}), 400

    score = definition_drift(model1, model2, word, reference_def, gemini_key)
    return jsonify({"drift_type": "definition", "score": score})

if __name__ == '__main__':
    app.run(debug=True)
