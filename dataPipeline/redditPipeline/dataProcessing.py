import spacy
import pandas as pd
import re
import requests
from collections import Counter
from nltk.corpus import words as nltk_words
import nltk
import time
from better_profanity import profanity

# Load NLTK words corpus
nltk.download('words')
from nltk.corpus import words
profanity.load_censor_words()

# Load Spacy English model
nlp = spacy.load("en_core_web_sm")

# Load the CSV file containing Reddit posts
df = pd.read_csv("reddit_trending_posts.csv")
texts = df['title'].fillna('') + ' ' + df['text'].fillna('')

# Combine all text into a single string and clean it
all_text = " ".join(texts).lower()
all_text = re.sub(r"[^a-zA-Z0-9\s]", "", all_text)

doc = nlp(all_text)

# Create a set of known words from NLTK's words corpus and Spacy's stop words
known_words = set(word.lower() for word in words.words())
stopwords = nlp.Defaults.stop_words

# Extract candidates: words that are alphabetic, longer than 2 characters, not in known words or stopwords
candidates = [
    token.text for token in doc
    if token.is_alpha
    and len(token.text) > 2
    and token.text.lower() not in known_words
    and token.text.lower() not in stopwords
]

# Count frequency
word_freq = Counter(candidates)
top_500 = word_freq.most_common(500)

# Function to fetch standard definitions from an API
def fetch_standard_definition(term):
    try:
        response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{term}")
        results = response.json()
        if isinstance(results, list):
            meanings = results[0].get("meanings", [])
            if meanings:
                defn = meanings[0].get("definitions", [{}])[0]
                return {
                    "definition": defn.get("definition", ""),
                    "example": defn.get("example", "")
                }
    except:
        return None
    return None

# Iterate and collect valid terms, skip profanity
verified_data = []

for word, freq in top_500:
    print(f"🔎 Looking up: {word}")
    result = fetch_standard_definition(word)
    
    if result:
        definition = result["definition"]
        example = result["example"]
    else:
        definition = "unknown"
        example = "unknown"

        # Profanity check
        if profanity.contains_profanity(word):
            continue
        if profanity.contains_profanity(definition):
            continue
        if profanity.contains_profanity(example or ""):
            continue

        verified_data.append({
            "word": word,
            "frequency": freq,
            "standard_definition": definition,
            "standard_example": example
        })
    # API rate limiting so my ip dont get blocked heheh
    time.sleep(0.5)  

df_out = pd.DataFrame(verified_data)
df_out.to_csv("standard_defined_terms_clean.csv", index=False)

