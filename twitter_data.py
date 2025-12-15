"""
Twitter Data Fetching Module
Fetches tweets using Twitter API v2
"""

import tweepy
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pandas as pd

# Load environment variables
load_dotenv()


def authenticate_twitter():
    """
    Authenticates with Twitter API using credentials from .env
    
    Returns:
        tweepy.Client: Authenticated Twitter API client
    """
    try:
        # Get credentials from environment variables
        bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        api_key = os.getenv('TWITTER_API_KEY')
        api_secret = os.getenv('TWITTER_API_SECRET')
        access_token = os.getenv('TWITTER_ACCESS_TOKEN')
        access_secret = os.getenv('TWITTER_ACCESS_SECRET')
        
        # Create client with API v2
        client = tweepy.Client(
            bearer_token=bearer_token,
            consumer_key=api_key,
            consumer_secret=api_secret,
            access_token=access_token,
            access_token_secret=access_secret,
            wait_on_rate_limit=True
        )
        
        print("✅ Successfully authenticated with Twitter API!")
        return client
        
    except Exception as e:
        print(f"❌ Error authenticating with Twitter: {e}")
        return None


def fetch_tweets(client, query, max_results=10):
    """
    Fetches recent tweets based on a search query
    
    Args:
        client: Authenticated tweepy.Client
        query: Search query string
        max_results: Maximum number of tweets to fetch (default 10, max 100 for recent search)
    
    Returns:
        pandas.DataFrame: DataFrame containing tweet data
    
    Raises:
        Exception: Re-raises exceptions (including 429 rate limit) for rotation handling
    """
    print(f"🔍 Searching for tweets with query: '{query}'")
    
    # Search recent tweets (let exceptions propagate for rotation)
    tweets = client.search_recent_tweets(
        query=query,
        max_results=max_results,
        tweet_fields=['created_at', 'author_id', 'public_metrics', 'lang', 'source'],
        expansions=['author_id'],
        user_fields=['username', 'name', 'verified']
    )
    
    if not tweets.data:
        print("⚠️ No tweets found!")
        return pd.DataFrame()
    
    # Process tweet data
    tweet_list = []
    users_dict = {user.id: user for user in tweets.includes.get('users', [])}
    
    for tweet in tweets.data:
        author = users_dict.get(tweet.author_id)
        
        tweet_data = {
            'tweet_id': tweet.id,
            'created_at': tweet.created_at,
            'text': tweet.text,
            'author_id': tweet.author_id,
            'author_username': author.username if author else 'Unknown',
            'author_name': author.name if author else 'Unknown',
            'verified': author.verified if author else False,
            'likes': tweet.public_metrics['like_count'],
            'retweets': tweet.public_metrics['retweet_count'],
            'replies': tweet.public_metrics['reply_count'],
            'language': tweet.lang,
            'source': tweet.source
        }
        tweet_list.append(tweet_data)
    
    df = pd.DataFrame(tweet_list)
    print(f"✅ Fetched {len(df)} tweets!")
    return df


def fetch_user_tweets(client, username, max_results=10):
    """
    Fetches recent tweets from a specific user
    
    Args:
        client: Authenticated tweepy.Client
        username: Twitter username (without @)
        max_results: Maximum number of tweets to fetch
    
    Returns:
        pandas.DataFrame: DataFrame containing tweet data
    """
    try:
        print(f"🔍 Fetching tweets from @{username}")
        
        # Get user ID
        user = client.get_user(username=username)
        if not user.data:
            print(f"⚠️ User @{username} not found!")
            return pd.DataFrame()
        
        user_id = user.data.id
        
        # Get user's tweets
        tweets = client.get_users_tweets(
            id=user_id,
            max_results=max_results,
            tweet_fields=['created_at', 'public_metrics', 'lang', 'source']
        )
        
        if not tweets.data:
            print("⚠️ No tweets found!")
            return pd.DataFrame()
        
        # Process tweet data
        tweet_list = []
        for tweet in tweets.data:
            tweet_data = {
                'tweet_id': tweet.id,
                'created_at': tweet.created_at,
                'text': tweet.text,
                'username': username,
                'likes': tweet.public_metrics['like_count'],
                'retweets': tweet.public_metrics['retweet_count'],
                'replies': tweet.public_metrics['reply_count'],
                'language': tweet.lang,
                'source': tweet.source
            }
            tweet_list.append(tweet_data)
        
        df = pd.DataFrame(tweet_list)
        print(f"✅ Fetched {len(df)} tweets from @{username}!")
        return df
        
    except Exception as e:
        print(f"❌ Error fetching user tweets: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    # Test the Twitter API connection
    print("Testing Twitter API connection...")
    client = authenticate_twitter()
    
    if client:
        # Test search
        df = fetch_tweets(client, query="Python programming", max_results=5)
        if not df.empty:
            print("\n📊 Sample tweets:")
            print(df[['created_at', 'author_username', 'text']].head())
