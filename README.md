# Trending Terms vs. Model Vocabulary Drift

## Project Overview
This project investigates how trending terms from social media (Reddit, Bluesky, etc.) are represented, tokenized, and understood by various large language models (LLMs) such as OpenAI GPT, Google Gemini, and Hugging Face models. The goal is to measure and visualize "vocabulary drift"—the gap between real-world language evolution and LLM vocabularies—using interactive metrics and visualizations.

- **Tokenization Drift:** How do different models split/tokenize new or trending words?
- **Semantic Drift:** How do models embed/understand these words?
- **Contextual Drift:** How do model-generated definitions/usages compare to real-world ones?

The project provides an interactive dashboard for exploring these drifts, as well as batch scripts for large-scale analysis.

---

## 📄 Reference PDF
For a detailed research overview, methodology, and results, see:
- [`Trending_Terms_vs_Model_Vocab_Drift.pdf`](./Trending_Terms_vs_Model_Vocab_Drift.pdf)

---

## 🎥 Project Video Walkthrough
Watch a video overview and demonstration of the dashboard and analysis:
- [YouTube: Trending Terms vs. Model Vocabulary Drift](https://youtu.be/sQcafKjbyzw)

---

## System Architecture

**1. Data Pipeline**
- Scrapes trending terms, definitions, and usage examples from Reddit, Bluesky, etc.
- Cleans and stores data in CSV files for further analysis.

**2. Metrics Calculation**
- `metricsCreation/metrics.py`: Core logic for tokenization, embeddings, and drift metrics (tokenization, semantic, contextual).
- Supports OpenAI GPT, Google Gemini, and Hugging Face models.

**3. API & Dashboard**
- `dashboard/app.py`: Flask backend serving drift/token/token count endpoints.
- `dashboard/templates/index.html`: Interactive UI for entering words, selecting models, and viewing results.
- Gemini API key support for Gemini models.

**4. Batch Data Metrics Update**
- `metricsCreation/updateDataMetrics.py`: Batch-updates cleaned data with drift metrics for large-scale analysis.

**5. Visualizations**
- `metricsCreation/visualizations.py`: Scripts for generating static plots (e.g., heatmaps, correlation analysis).

---

## How to Run the Dashboard

1. **Install dependencies:**
   ```bash
   cd dashboard
   pip install -r requirements.txt
   ```

2. **Start the dashboard:**
   ```bash
   python run.py --port 5050
   ```
   (You can use any port; 5050 is just an example.)

3. **Open your browser:**
   - Go to [http://localhost:5050](http://localhost:5050)

4. **Usage:**
   - Enter any word or select from trending terms.
   - Select any two models (OpenAI, Gemini, Hugging Face) for comparison.
   - Enter your Gemini API key if using Gemini models.
   - View drift scores and token counts in real time.

---

## Main Components
- **dashboard/**: Interactive dashboard (Flask backend + HTML/JS frontend)
- **metricsCreation/**: Core drift metrics, batch update scripts, and visualizations
- **dataPipeline/**: Social media scraping and cleaning scripts

---

## Citation & Links
- [YouTube Project Video](https://youtu.be/sQcafKjbyzw)
- [Trending_Terms_vs_Model_Vocab_Drift.pdf](./Trending_Terms_vs_Model_Vocab_Drift.pdf)

---

## Acknowledgments
- OpenAI, Google, and Hugging Face for model APIs and libraries
- Reddit, Bluesky, and other platforms for trending term data

---

For questions or contributions, please open an issue or submit a pull request.
