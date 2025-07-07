# Drift Analysis Dashboard

A Flask-based web dashboard for analyzing vocabulary drift between different language models. This dashboard allows you to compare how different models tokenize and semantically represent trending terms.

## Features

- **Tokenization Drift Analysis**: Compare how different models tokenize the same terms
- **Semantic Drift Analysis**: Compare semantic representations of terms across models
- **Token Counting**: Get token counts for terms using different models
- **Modern Web Interface**: Clean, responsive UI with real-time results
- **Extensible Architecture**: Easy to add new models and analysis types

## Supported Models

### OpenAI Models (via tiktoken)
- gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo

### Hugging Face Models (via transformers)
- gpt2, bert-base-uncased, roberta-base, distilbert-base-uncased
- t5-small, facebook/bart-base, albert-base-v2

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify installation:**
   ```bash
   python run.py
   ```

## Usage

1. **Start the dashboard:**
   ```bash
   cd dashboard
   python run.py
   ```

2. **Access the dashboard:**
   - Open your browser and go to: http://localhost:5000
   - The API endpoints are available at: http://localhost:5000/api/

3. **Using the dashboard:**
   - Select a trending term from the available options
   - Choose two models to compare
   - Run tokenization or semantic drift analysis
   - View results in real-time

## API Endpoints

### GET /api/terms
Get all available trending terms.

### GET /api/models  
Get all available models for analysis.

### POST /api/drift/tokenization
Compute tokenization drift between two models.

**Request body:**
```json
{
  "word": "rizz",
  "model1": "gpt2", 
  "model2": "bert-base-uncased"
}
```

### POST /api/drift/semantic
Compute semantic drift between two models.

**Request body:**
```json
{
  "word": "rizz",
  "model1": "gpt2",
  "model2": "bert-base-uncased" 
}
```

### GET /api/tokens/{term}?model={model}
Get token count for a term using a specific model.

### GET /api/health
Health check endpoint.

## Architecture

The dashboard uses:
- **Flask**: Web framework for the backend API
- **Flask-CORS**: Cross-origin resource sharing support
- **Transformers**: Hugging Face models and tokenizers
- **Tiktoken**: OpenAI tokenizer support
- **Sentence Transformers**: Semantic similarity calculations

## Extending the Dashboard

### Adding New Models
1. Add the model to the `AVAILABLE_MODELS` dictionary in `app.py`
2. Ensure the model is supported by the metrics functions

### Adding New Analysis Types
1. Create a new endpoint in `app.py`
2. Add corresponding frontend functionality in `templates/index.html`
3. Implement the analysis logic in `metricsCreation/metrics.py`

### Adding API Key Support
The dashboard is designed to be easily extended with API key support:
- Add API key fields to the frontend
- Modify the drift functions to accept API keys
- Update the backend endpoints to handle API keys

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install missing dependencies with `pip install -r requirements.txt`

2. **Model loading errors**: Some models require internet connection for first-time download

3. **Memory issues**: Large models may require significant RAM

### Debug Mode
The dashboard runs in debug mode by default. Check the console for detailed error messages.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the Trending Terms vs. Model Vocabulary Drift analysis. 