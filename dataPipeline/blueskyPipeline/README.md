# Bluesky Data Pipeline 📊

This project scrapes high-engagement posts from Bluesky, processes the text data, and extracts new or uncommon terms with their definitions. It consists of two main scripts:

- **blueskyScrape.py**: Scrapes high-engagement Bluesky posts and saves them to a CSV file.
- **dataProcessing.py**: Processes the scraped data, filters out common/profane words, and fetches definitions for new terms.

---

## Bluesky API Key Setup Instructions 
To use the Bluesky API, you need credentials from Bluesky:

1. Go to [Bluesky](https://bsky.app) and create an account.
2. Go to Settings → App passwords.
3. Create a new app password.
4. You will need:
   - **handle**: Your Bluesky handle (e.g., "username.bsky.social")
   - **app_password**: The app password you created

Create a `.env` file in the project directory with the following content:

```
BLUESKY_HANDLE=your_handle_here
BLUESKY_APP_PASSWORD=your_app_password_here
```

**Do not share your credentials publicly.**

---

## Running the Pipeline

#### a. Scrape High-Engagement Bluesky Posts

```bash
python blueskyScrape.py
```

This will create a `bluesky_trending_posts.csv` file.

This is all the raw unfiltered data from the top 1000 high-engagement posts:
- Scrapes posts from the timeline feed (popular posts)
- Extracts engagement metrics (likes, reposts, replies)
- Sorts by total engagement (highest first)
- Focuses on viral content where slang spreads

#### b. Process the Data

```bash
python dataProcessing.py
```

This will create a `standard_defined_terms_clean.csv` file with filtered terms with their frequency, definition, and example in sentence.

---

## File Overview

### blueskyScrape.py

- Loads Bluesky API credentials from `.env`
- Scrapes the top 1000 high-engagement posts from timeline feed
- Extracts engagement metrics (likes, reposts, replies)
- Saves post data (text, author, timestamp, engagement) to `bluesky_trending_posts.csv`

### dataProcessing.py

- Reads `bluesky_trending_posts.csv`
- Cleans and tokenizes text
- Filters out common English words and stopwords
- Checks for profanity
- Fetches definitions for new terms using a dictionary API
- Saves results to `standard_defined_terms_clean.csv`

---
