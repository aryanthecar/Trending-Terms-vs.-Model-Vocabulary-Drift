import pandas as pd
import os
from dotenv import load_dotenv
from atproto import Client

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
    print(f" Login failed: {e}")
    exit(1)

print(f"\nFetching top {POST_LIMIT} high-engagement posts...")

try:
    # Get timeline feed (contains popular posts)
    cursor = None
    posts_collected = 0
    
    while posts_collected < POST_LIMIT:
        # Get batch of posts from timeline
        limit = min(100, POST_LIMIT - posts_collected)
        timeline = client.get_timeline(cursor=cursor, limit=limit)
        
        if not timeline.feed:
            print("No more posts in timeline.")
            break
        
        for feed_view in timeline.feed:
            if posts_collected >= POST_LIMIT:
                break
                
            post = feed_view.post
            
            # Extract engagement metrics
            like_count = feed_view.post.likeCount if hasattr(feed_view.post, 'likeCount') else 0
            repost_count = feed_view.post.repostCount if hasattr(feed_view.post, 'repostCount') else 0
            reply_count = feed_view.post.replyCount if hasattr(feed_view.post, 'replyCount') else 0
            
            # Calculate total engagement
            total_engagement = like_count + repost_count + reply_count
            
            posts.append({
                "search_term": "high_engagement",  # Keep same structure as Reddit
                "title": "",  # Bluesky posts don't have titles like Reddit
                "text": post.record.text if hasattr(post.record, 'text') else "",
                "score": total_engagement,  # Use engagement as score
                "created_utc": post.record.created_at if hasattr(post.record, 'created_at') else "",
                "id": post.cid,
                "url": post.uri,
                "author_handle": post.author.handle if hasattr(post.author, 'handle') else "",
                "like_count": like_count,
                "repost_count": repost_count,
                "reply_count": reply_count,
                "total_engagement": total_engagement
            })
            
            posts_collected += 1
            
            if posts_collected % 50 == 0:
                print(f"Collected {posts_collected} posts... (engagement: {total_engagement})")
        
        cursor = timeline.cursor
        
except Exception as e:
    print(f" Error fetching posts: {e}")

# Sort by engagement (highest first)
df = pd.DataFrame(posts)

if len(posts) == 0:
    print("No posts found in timeline. Trying search approach...")
    
    # Fallback: use search terms to get posts
    search_terms = ["trending", "viral", "meme", "slang", "internet"]
    
    for term in search_terms:
        try:
            results = client.app.bsky.feed.search_posts({"q": term, "limit": 100})
            
            for post in results.posts:
                posts.append({
                    "search_term": term,
                    "title": "",
                    "text": post.record.text if hasattr(post.record, 'text') else "",
                    "score": 0,
                    "created_utc": post.record.created_at if hasattr(post.record, 'created_at') else "",
                    "id": post.cid,
                    "url": post.uri,
                    "author_handle": post.author.handle if hasattr(post.author, 'handle') else "",
                    "like_count": 0,
                    "repost_count": 0,
                    "reply_count": 0,
                    "total_engagement": 0
                })
                
        except Exception as e:
            print(f"Error searching for '{term}': {e}")
    
    df = pd.DataFrame(posts)

if len(posts) > 0:
    df = df.sort_values('total_engagement', ascending=False)
    df.to_csv("bluesky_trending_posts.csv", index=False)
    print(f"\n✅ Saved {len(posts)} posts to bluesky_trending_posts.csv")
    print(f"Top engagement post: {df.iloc[0]['total_engagement']} interactions")
    print(f"Average engagement: {df['total_engagement'].mean():.1f} interactions")
else:
    print("❌ No posts could be collected") 