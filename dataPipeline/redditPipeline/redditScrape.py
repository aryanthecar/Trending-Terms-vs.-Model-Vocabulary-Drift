import praw
import pandas as pd
import os
from dotenv import load_dotenv

# Set up Reddit API credentials
load_dotenv()  # Load environment variables from .env file
id = os.getenv("CLIENT_ID")
secret = os.getenv("CLIENT_SECRET")
user = os.getenv("USER_AGENT")

reddit = praw.Reddit(
    client_id=id,
    client_secret=secret,
    user_agent=user
)

# List of subreddits to extract posts from
subreddits = ["teenagers", "OutOfTheLoop", "linguistics", "AskReddit", "dankmemes", "brainrot", "SlangExplained"]

# How many posts to pull per subreddit
POST_LIMIT = 50

# Store results
posts = []

# Scrape posts
for sub in subreddits:
    subreddit = reddit.subreddit(sub)
    print(f"\nFetching posts from r/{sub}...")

    for post in subreddit.hot(limit=POST_LIMIT):
        posts.append({
            "subreddit": sub,
            "title": post.title,
            "text": post.selftext,
            "score": post.score,
            "created_utc": post.created_utc,
            "id": post.id,
            "url": post.url,
        })
        print(f"📝 {post.title}")

# Store in CSV
df = pd.DataFrame(posts)
df.to_csv("reddit_trending_posts.csv", index=False)
