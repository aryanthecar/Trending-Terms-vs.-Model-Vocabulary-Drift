import pandas as pd
import os
from dotenv import load_dotenv
from atproto import Client
import time

# Set up Bluesky API credentials
load_dotenv()  # Load environment variables from .env file
handle = os.getenv("BLUESKY_HANDLE")
password = os.getenv("BLUESKY_APP_PASSWORD")

# Initialize Bluesky client
client = Client()

# How many posts to pull (high engagement posts)
POST_LIMIT = 1000

# Store results
posts = []

# Login to Bluesky
try:
    client.login(handle, password)
    print(f"✅ Logged in as {handle}")
except Exception as e:
    print(f"❌ Login failed: {e}")
    exit(1)

print(f"\nFetching top {POST_LIMIT} high-engagement posts...")

# Use search to find popular posts
search_terms = [
    "",  # Empty search to get recent posts
    "trending",
    "viral", 
    "popular",
    "meme",
    "funny",
    "news",
    "tech",
    "science"
]

for term in search_terms:
    if len(posts) >= POST_LIMIT:
        break
        
    print(f"\nSearching for posts with term: '{term if term else 'recent posts'}'...")
    
    try:
        # Search for posts
        search_params = {"limit": 100}
        if term:
            search_params["q"] = term
            
        search_results = client.app.bsky.feed.search_posts(search_params)
        
        if not search_results.posts:
            print(f"No posts found for term: {term}")
            continue
            
        for post in search_results.posts:
            if len(posts) >= POST_LIMIT:
                break
                
            # Extract engagement metrics if available
            like_count = getattr(post, 'likeCount', 0)
            repost_count = getattr(post, 'repostCount', 0)
            reply_count = getattr(post, 'replyCount', 0)
            
            # Calculate total engagement
            total_engagement = like_count + repost_count + reply_count
            
            posts.append({
                "source": f"search_{term}" if term else "recent_posts",
                "title": "",
                "text": getattr(post.record, 'text', ''),
                "score": total_engagement,
                "created_utc": getattr(post.record, 'created_at', ''),
                "id": post.cid,
                "url": post.uri,
                "author_handle": getattr(post.author, 'handle', ''),
                "like_count": like_count,
                "repost_count": repost_count,
                "reply_count": reply_count,
                "total_engagement": total_engagement
            })
            
        print(f"✅ Collected {len([p for p in posts if p['source'] == (f'search_{term}' if term else 'recent_posts')])} posts from search term: {term}")
        
        # Rate limiting
        time.sleep(1)
        
    except Exception as e:
        print(f"❌ Error searching for term '{term}': {e}")

if len(posts) > 0:
    df = pd.DataFrame(posts)
    df = df.sort_values('total_engagement', ascending=False)
    df.to_csv("bluesky_trending_posts.csv", index=False)
    print(f"\n✅ Saved {len(posts)} posts to bluesky_trending_posts.csv")
    print(f"Top engagement post: {df.iloc[0]['total_engagement']} interactions")
    print(f"Average engagement: {df['total_engagement'].mean():.1f} interactions")
    print(f"Source breakdown:")
    print(df['source'].value_counts())
else:
    print("❌ No posts could be collected from any search") 