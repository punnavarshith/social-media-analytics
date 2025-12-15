"""
Reddit Data Fetching Module
Fetches posts and comments from Reddit using PRAW (Python Reddit API Wrapper)
"""

import praw
import os
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd

# Load environment variables
load_dotenv()


def authenticate_reddit():
    """
    Authenticates with Reddit API using credentials from .env
    
    Returns:
        praw.Reddit: Authenticated Reddit API client
    """
    try:
        # Get credentials from environment variables
        client_id = os.getenv('REDDIT_CLIENT_ID')
        client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        user_agent = os.getenv('REDDIT_USER_AGENT', 'python:social_data_collector:v1.0')
        
        # Create Reddit instance
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        print("✅ Successfully authenticated with Reddit API!")
        print(f"📊 Read-only mode: {reddit.read_only}")
        return reddit
        
    except Exception as e:
        print(f"❌ Error authenticating with Reddit: {e}")
        return None


def fetch_subreddit_posts(reddit, subreddit_name, sort_by='hot', limit=10, time_filter='day'):
    """
    Fetches posts from a specific subreddit
    
    Args:
        reddit: Authenticated praw.Reddit instance
        subreddit_name: Name of the subreddit (without r/)
        sort_by: How to sort posts ('hot', 'new', 'top', 'rising', 'controversial')
        limit: Maximum number of posts to fetch (default 10)
        time_filter: Time filter for 'top' and 'controversial' ('hour', 'day', 'week', 'month', 'year', 'all')
    
    Returns:
        pandas.DataFrame: DataFrame containing Reddit post data
    """
    try:
        print(f"🔍 Fetching {sort_by} posts from r/{subreddit_name}...")
        
        # Get subreddit
        subreddit = reddit.subreddit(subreddit_name)
        
        # Fetch posts based on sort method
        if sort_by == 'hot':
            posts = subreddit.hot(limit=limit)
        elif sort_by == 'new':
            posts = subreddit.new(limit=limit)
        elif sort_by == 'top':
            posts = subreddit.top(time_filter=time_filter, limit=limit)
        elif sort_by == 'rising':
            posts = subreddit.rising(limit=limit)
        elif sort_by == 'controversial':
            posts = subreddit.controversial(time_filter=time_filter, limit=limit)
        else:
            print(f"⚠️ Invalid sort_by: {sort_by}. Using 'hot' instead.")
            posts = subreddit.hot(limit=limit)
        
        # Process post data
        post_list = []
        for post in posts:
            post_data = {
                'post_id': post.id,
                'subreddit': subreddit_name,
                'title': post.title,
                'author': str(post.author) if post.author else '[deleted]',
                'created_at': datetime.fromtimestamp(post.created_utc),
                'score': post.score,
                'upvote_ratio': post.upvote_ratio,
                'num_comments': post.num_comments,
                'url': post.url,
                'permalink': f"https://reddit.com{post.permalink}",
                'is_self': post.is_self,
                'selftext': post.selftext if post.is_self else '',
                'link_flair_text': post.link_flair_text,
                'over_18': post.over_18,
                'spoiler': post.spoiler,
                'stickied': post.stickied
            }
            post_list.append(post_data)
        
        df = pd.DataFrame(post_list)
        print(f"✅ Fetched {len(df)} posts from r/{subreddit_name}!")
        return df
        
    except Exception as e:
        print(f"❌ Error fetching subreddit posts: {e}")
        return pd.DataFrame()


def fetch_post_comments(reddit, post_id, limit=10):
    """
    Fetches top comments from a specific Reddit post
    
    Args:
        reddit: Authenticated praw.Reddit instance
        post_id: Reddit post ID
        limit: Maximum number of top-level comments to fetch
    
    Returns:
        pandas.DataFrame: DataFrame containing comment data
    """
    try:
        print(f"🔍 Fetching comments from post {post_id}...")
        
        # Get submission
        submission = reddit.submission(id=post_id)
        
        # Fetch comments
        submission.comments.replace_more(limit=0)  # Remove "MoreComments" instances
        comments = submission.comments.list()[:limit]
        
        # Process comment data
        comment_list = []
        for comment in comments:
            if hasattr(comment, 'body'):  # Make sure it's a Comment object
                comment_data = {
                    'comment_id': comment.id,
                    'post_id': post_id,
                    'author': str(comment.author) if comment.author else '[deleted]',
                    'body': comment.body,
                    'created_utc': datetime.fromtimestamp(comment.created_utc),
                    'score': comment.score,
                    'is_submitter': comment.is_submitter,
                    'permalink': f"https://reddit.com{comment.permalink}"
                }
                comment_list.append(comment_data)
        
        df = pd.DataFrame(comment_list)
        print(f"✅ Fetched {len(df)} comments!")
        return df
        
    except Exception as e:
        print(f"❌ Error fetching comments: {e}")
        return pd.DataFrame()


def search_reddit(reddit, query, sort='relevance', time_filter='all', limit=10):
    """
    Searches Reddit for posts matching a query
    
    Args:
        reddit: Authenticated praw.Reddit instance
        query: Search query string
        sort: How to sort results ('relevance', 'hot', 'top', 'new', 'comments')
        time_filter: Time filter ('hour', 'day', 'week', 'month', 'year', 'all')
        limit: Maximum number of results
    
    Returns:
        pandas.DataFrame: DataFrame containing search results
    """
    try:
        print(f"🔍 Searching Reddit for: '{query}'...")
        
        # Search all of Reddit
        posts = reddit.subreddit('all').search(
            query=query,
            sort=sort,
            time_filter=time_filter,
            limit=limit
        )
        
        # Process results
        post_list = []
        for post in posts:
            post_data = {
                'post_id': post.id,
                'subreddit': post.subreddit.display_name,
                'title': post.title,
                'author': str(post.author) if post.author else '[deleted]',
                'created_utc': datetime.fromtimestamp(post.created_utc),
                'score': post.score,
                'upvote_ratio': post.upvote_ratio,
                'num_comments': post.num_comments,
                'url': post.url,
                'permalink': f"https://reddit.com{post.permalink}",
                'is_self': post.is_self,
                'selftext': post.selftext[:500] if post.is_self else ''  # Limit selftext length
            }
            post_list.append(post_data)
        
        df = pd.DataFrame(post_list)
        print(f"✅ Found {len(df)} posts matching '{query}'!")
        return df
        
    except Exception as e:
        print(f"❌ Error searching Reddit: {e}")
        return pd.DataFrame()


def get_trending_subreddits(reddit, limit=10):
    """
    Fetches trending/popular subreddits
    
    Args:
        reddit: Authenticated praw.Reddit instance
        limit: Number of subreddits to fetch
    
    Returns:
        pandas.DataFrame: DataFrame containing subreddit data
    """
    try:
        print(f"🔍 Fetching trending subreddits...")
        
        # Get popular subreddits
        subreddits = reddit.subreddits.popular(limit=limit)
        
        # Process subreddit data
        subreddit_list = []
        for subreddit in subreddits:
            subreddit_data = {
                'subreddit_name': subreddit.display_name,
                'subscribers': subreddit.subscribers,
                'description': subreddit.public_description[:200],  # Limit description
                'created_utc': datetime.fromtimestamp(subreddit.created_utc),
                'over_18': subreddit.over18,
                'url': f"https://reddit.com/r/{subreddit.display_name}"
            }
            subreddit_list.append(subreddit_data)
        
        df = pd.DataFrame(subreddit_list)
        print(f"✅ Fetched {len(df)} trending subreddits!")
        return df
        
    except Exception as e:
        print(f"❌ Error fetching trending subreddits: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # Test the Reddit API connection
    print("Testing Reddit API connection...")
    reddit = authenticate_reddit()
    
    if reddit:
        # Test 1: Fetch posts from a subreddit
        print("\n" + "="*60)
        print("TEST 1: Fetching posts from r/Python")
        print("="*60)
        df_posts = fetch_subreddit_posts(reddit, 'Python', sort_by='hot', limit=5)
        if not df_posts.empty:
            print("\n📊 Sample posts:")
            print(df_posts[['title', 'score', 'num_comments', 'author']].head())
        
        # Test 2: Search Reddit
        print("\n" + "="*60)
        print("TEST 2: Searching for 'artificial intelligence'")
        print("="*60)
        df_search = search_reddit(reddit, query='artificial intelligence', limit=5)
        if not df_search.empty:
            print("\n📊 Search results:")
            print(df_search[['subreddit', 'title', 'score']].head())
        
        # Test 3: Get trending subreddits
        print("\n" + "="*60)
        print("TEST 3: Fetching trending subreddits")
        print("="*60)
        df_trending = get_trending_subreddits(reddit, limit=5)
        if not df_trending.empty:
            print("\n📊 Trending subreddits:")
            print(df_trending[['subreddit_name', 'subscribers']].head())
