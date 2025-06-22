# Reddit Data Pipeline 📊

This project scrapes trending posts from selected subreddits using the Reddit API, processes the text data, and extracts new or uncommon terms with their definitions. It consists of two main scripts:

- **redditScrape.py**: Scrapes Reddit posts and saves them to a CSV file.
- **dataProcessing.py**: Processes the scraped data, filters out common/profane words, and fetches definitions for new terms.

---

## Reddit API Key Setup Instructions 😱
To use the Reddit API, you need credentials from Reddit:

1. Go to [Reddit Apps](https://www.reddit.com/prefs/apps).
2. Click **"create another app"** at the bottom.
3. Fill in:
   - **name**: Any name you like
   - **type**: Script
   - **redirect uri**: http://localhost:8080 (can be anything for scripts)
4. After creation, you will see:
   - **client_id** (this will be emailed to you)
   - **client_secret**
   - **user_agent** (any descriptive string, e.g., `my_reddit_app:v1.0 (by /u/yourusername)`)

Create a `.env` file in the project directory with the following content:

```
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=your_user_agent_here
```

**Do not share your credentials publicly.**

---

## Running the Pipeline

#### a. Scrape Reddit Posts

```bash
python redditScrape.py
```

This will create a `reddit_trending_posts.csv` file.

This is all the raw unfiltered data from the following subreddits:
1. teenagers
2. OutOfTheLoop
3. linguistics
4. AskReddit
5. dankmemes
6. brainrot
7. SlangExplained

#### b. Process the Data

```bash
python dataProcessing.py
```

This will create a `standard_defined_terms_clean.csv` file with filtered terms with their frequency, definition, and example in sentence.

---

## File Overview

### redditScrape.py

- Loads Reddit API credentials from `.env`
- Scrapes the latest hot posts from a list of subreddits
- Saves post data (title, text, score, etc.) to `reddit_trending_posts.csv`

### dataProcessing.py

- Reads `reddit_trending_posts.csv`
- Cleans and tokenizes text
- Filters out common English words and stopwords
- Checks for profanity
- Fetches definitions for new terms using a dictionary API
- Saves results to `standard_defined_terms_clean.csv`

---
