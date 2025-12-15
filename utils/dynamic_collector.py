"""
Dynamic Topic-Based Data Collector
Collects social media data for ANY user-provided topic
"""

import praw
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config.api_keys import REDDIT_CONFIG, YOUTUBE_API_KEY, YOUTUBE_ENABLED
except ImportError:
    print("⚠️ Config file not found. Using default settings.")
    REDDIT_CONFIG = {}
    YOUTUBE_API_KEY = ""
    YOUTUBE_ENABLED = False

# Import upload modules
try:
    from google_sheet_connect import connect_to_google_sheets, get_sheet
    GOOGLE_SHEETS_AVAILABLE = True
except ImportError:
    GOOGLE_SHEETS_AVAILABLE = False
    print("⚠️ Google Sheets module not available")

try:
    from utils.supabase_db import SupabaseDB
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("⚠️ Supabase module not available")


class DynamicTopicCollector:
    """
    Collects social media data for ANY user-provided topic
    Supports: Reddit (required), YouTube (optional)
    """
    
    def __init__(self):
        self.reddit_client = None
        self.youtube_client = None
        self.setup_clients()
    
    def setup_clients(self):
        """Initialize API clients"""
        try:
            # Reddit client (required)
            if all(REDDIT_CONFIG.values()):
                self.reddit_client = praw.Reddit(
                    client_id=REDDIT_CONFIG['client_id'],
                    client_secret=REDDIT_CONFIG['client_secret'],
                    user_agent=REDDIT_CONFIG['user_agent']
                )
                print("✅ Reddit client initialized")
            else:
                print("⚠️ Reddit credentials not configured")
            
            # YouTube client (optional)
            if YOUTUBE_ENABLED and YOUTUBE_API_KEY:
                try:
                    from googleapiclient.discovery import build  # type: ignore
                    self.youtube_client = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
                    print("✅ YouTube client initialized")
                except ImportError:
                    print("⚠️ Google API client not installed. Install with: pip install google-api-python-client")
                    self.youtube_client = None
                except Exception as e:
                    print(f"⚠️ YouTube client error: {e}")
                    self.youtube_client = None
            
        except Exception as e:
            print(f"⚠️ Error initializing clients: {e}")
    
    def _generate_search_queries(self, topic: str):
        """
        Generate search variations for any topic
        
        Examples:
        Input: "Nike shoes" 
        Output: ["Nike shoes", "Nike shoes review", "best Nike shoes", ...]
        
        Input: "iPhone 15"
        Output: ["iPhone 15", "iPhone 15 review", "iPhone 15 vs", ...]
        """
        base_queries = [
            topic,
            f"{topic} review",
            f"{topic} vs",
            f"best {topic}",
            f"{topic} worth it",
            f"{topic} comparison",
            f"{topic} experience",
            f"{topic} recommendation",
            f"{topic} opinion"
        ]
        
        return base_queries
    
    def collect_reddit_data(self, topic: str, limit: int = 100):
        """
        Collect Reddit posts for ANY user-provided topic
        
        Args:
            topic: User's topic (e.g., "Nike shoes", "iPhone 15", "Starbucks")
            limit: Max posts to collect (distributed across search queries)
        
        Returns:
            DataFrame with Reddit posts about the topic
        """
        if not self.reddit_client:
            print("❌ Reddit client not initialized. Please configure API credentials.")
            return pd.DataFrame()
        
        print(f"\n📡 Collecting Reddit data for topic: '{topic}'")
        
        # Generate search variations
        search_queries = self._generate_search_queries(topic)
        posts_per_query = max(10, limit // len(search_queries))
        
        all_posts = []
        
        for query in search_queries:
            try:
                print(f"  🔍 Searching: '{query}'...")
                
                # Search across ALL of Reddit for this specific topic
                for submission in self.reddit_client.subreddit('all').search(
                    query,
                    limit=posts_per_query,
                    time_filter='year',  # Last year only
                    sort='relevance'
                ):
                    post_data = {
                        'platform': 'reddit',
                        'post_id': submission.id,
                        'title': submission.title,
                        'selftext': submission.selftext,
                        'clean_title': submission.title,
                        'author': str(submission.author),
                        'subreddit': submission.subreddit.display_name,
                        'created_at': datetime.utcfromtimestamp(submission.created_utc).isoformat() + 'Z',
                        'score': submission.score,
                        'num_comments': submission.num_comments,
                        'upvote_ratio': submission.upvote_ratio,
                        'url': submission.url,
                        'permalink': f"https://reddit.com{submission.permalink}",
                        'is_self': submission.is_self,
                        'search_query': query,
                        'topic': topic,
                        'collected_at': datetime.utcnow().isoformat() + 'Z'
                    }
                    all_posts.append(post_data)
                
                # Rate limit safety
                time.sleep(2)
                
            except Exception as e:
                print(f"  ⚠️ Error searching '{query}': {e}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(all_posts)
        
        # Remove duplicates (same post from different queries)
        if not df.empty:
            df = df.drop_duplicates(subset=['post_id'])
            print(f"\n✅ Collected {len(df)} unique Reddit posts about '{topic}'")
        else:
            print(f"\n⚠️ No Reddit posts found for '{topic}'")
        
        return df
    
    def collect_youtube_data(self, topic: str, limit: int = 50):
        """
        Collect YouTube videos for ANY user-provided topic (Optional)
        
        Args:
            topic: User's topic
            limit: Max videos to collect
        
        Returns:
            DataFrame with YouTube videos about the topic
        """
        if not self.youtube_client:
            print("⚠️ YouTube client not available. Skipping YouTube collection.")
            return pd.DataFrame()
        
        print(f"\n📺 Collecting YouTube data for topic: '{topic}'")
        
        search_queries = self._generate_search_queries(topic)
        videos_per_query = max(5, limit // len(search_queries))
        
        all_videos = []
        
        for query in search_queries:
            try:
                print(f"  🔍 Searching: '{query}'...")
                
                from googleapiclient.errors import HttpError  # type: ignore
                
                # Search videos
                search_response = self.youtube_client.search().list(
                    q=query,
                    part='id,snippet',
                    maxResults=videos_per_query,
                    order='relevance',
                    type='video',
                    publishedAfter=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%dT%H:%M:%SZ')
                ).execute()
                
                for item in search_response.get('items', []):
                    video_id = item['id']['videoId']
                    snippet = item['snippet']
                    
                    # Get video statistics
                    video_response = self.youtube_client.videos().list(
                        part='statistics',
                        id=video_id
                    ).execute()
                    
                    stats = video_response['items'][0]['statistics'] if video_response['items'] else {}
                    
                    video_data = {
                        'platform': 'youtube',
                        'video_id': video_id,
                        'title': snippet['title'],
                        'description': snippet['description'],
                        'channel': snippet['channelTitle'],
                        'published_at': snippet['publishedAt'],
                        'view_count': int(stats.get('viewCount', 0)),
                        'like_count': int(stats.get('likeCount', 0)),
                        'comment_count': int(stats.get('commentCount', 0)),
                        'url': f"https://www.youtube.com/watch?v={video_id}",
                        'search_query': query,
                        'topic': topic,
                        'collected_at': datetime.utcnow().isoformat() + 'Z'
                    }
                    all_videos.append(video_data)
                
                time.sleep(1)
                
            except Exception as e:
                print(f"  ⚠️ YouTube error for '{query}': {e}")
                continue
        
        df = pd.DataFrame(all_videos)
        
        if not df.empty:
            df = df.drop_duplicates(subset=['video_id'])
            print(f"\n✅ Collected {len(df)} unique YouTube videos about '{topic}'")
        else:
            print(f"\n⚠️ No YouTube videos found for '{topic}'")
        
        return df
    
    def collect_for_topic(self, topic: str, reddit_limit: int = 100, youtube_limit: int = 50, save_local: bool = False):
        """
        Collect data from all available platforms for a specific topic
        
        Args:
            topic: The product/service/brand to analyze
                   Examples: "Nike shoes", "iPhone 15", "Starbucks coffee"
            reddit_limit: Max Reddit posts to collect
            youtube_limit: Max YouTube videos to collect
            save_local: If True, save CSV files locally (default: False - direct upload only)
        
        Returns:
            dict: {'reddit': df, 'youtube': df, 'topic': str, 'collected_at': datetime}
        """
        print("\n" + "="*60)
        print(f"🚀 DYNAMIC TOPIC DATA COLLECTION")
        print(f"Topic: {topic}")
        print("="*60)
        
        # Collect from Reddit (primary source)
        reddit_df = self.collect_reddit_data(topic, reddit_limit)
        
        # Collect from YouTube (optional)
        youtube_df = self.collect_youtube_data(topic, youtube_limit)
        
        # Optional: Save to CSV files (disabled by default - direct upload preferred)
        if save_local:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            topic_slug = topic.lower().replace(' ', '_').replace('/', '_')
            
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            if not reddit_df.empty:
                filepath = f'data/{topic_slug}_reddit_{timestamp}.csv'
                reddit_df.to_csv(filepath, index=False)
                print(f"\n💾 Saved Reddit data locally: {filepath}")
            
            if not youtube_df.empty:
                filepath = f'data/{topic_slug}_youtube_{timestamp}.csv'
                youtube_df.to_csv(filepath, index=False)
                print(f"💾 Saved YouTube data locally: {filepath}")
        else:
            print(f"\n📤 Skipping local CSV save (direct upload mode)")
        
        # Upload to Google Sheets and Supabase (direct upload workflow)
        if not reddit_df.empty:
            self._upload_to_storage(reddit_df, topic)
        
        # Combine Reddit and YouTube data into single DataFrame
        combined_dfs = []
        if not reddit_df.empty:
            combined_dfs.append(reddit_df)
        if not youtube_df.empty:
            combined_dfs.append(youtube_df)
        
        combined_df = pd.concat(combined_dfs, ignore_index=True) if combined_dfs else pd.DataFrame()
        
        print("\n" + "="*60)
        print("✅ DATA COLLECTION COMPLETE!")
        print("="*60)
        
        return {
            'reddit': reddit_df,
            'youtube': youtube_df,
            'combined_df': combined_df,  # Add combined DataFrame
            'topic': topic,
            'collected_at': datetime.now(),
            'total_posts': len(reddit_df) + len(youtube_df)
        }
    
    def get_topic_summary(self, data: dict):
        """
        Generate a summary of collected topic data
        
        Args:
            data: Output from collect_for_topic()
        
        Returns:
            dict: Summary statistics and insights
        """
        reddit_df = data['reddit']
        youtube_df = data['youtube']
        
        summary = {
            'topic': data['topic'],
            'collected_at': data['collected_at'],
            'total_posts': data['total_posts'],
            'reddit_posts': len(reddit_df),
            'youtube_videos': len(youtube_df)
        }
        
        # Reddit insights
        if not reddit_df.empty:
            summary['reddit_insights'] = {
                'avg_score': reddit_df['score'].mean(),
                'total_comments': reddit_df['num_comments'].sum(),
                'top_subreddits': reddit_df['subreddit'].value_counts().head(5).to_dict(),
                'avg_upvote_ratio': reddit_df['upvote_ratio'].mean()
            }
        
        # YouTube insights
        if not youtube_df.empty:
            summary['youtube_insights'] = {
                'total_views': youtube_df['view_count'].sum(),
                'avg_views': youtube_df['view_count'].mean(),
                'total_likes': youtube_df['like_count'].sum(),
                'total_comments': youtube_df['comment_count'].sum()
            }
        
        return summary
    
    def _upload_to_storage(self, df: pd.DataFrame, topic: str):
        """
        Upload collected data to Google Sheets and Supabase
        
        Args:
            df: DataFrame with collected posts
            topic: The topic being collected
        """
        print(f"\n📤 Uploading to Google Sheets & Supabase...")
        
        # Prepare data
        upload_df = df.copy()
        
        # Ensure topic column exists
        if 'topic' not in upload_df.columns:
            upload_df['topic'] = topic
        
        # Convert timestamps to strings for Google Sheets compatibility
        if 'created_at' in upload_df.columns:
            upload_df['created_at'] = pd.to_datetime(upload_df['created_at'], errors='coerce', utc=True)
            upload_df['created_at'] = upload_df['created_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Convert any remaining datetime/timestamp columns to strings
        for col in upload_df.columns:
            if pd.api.types.is_datetime64_any_dtype(upload_df[col]):
                upload_df[col] = upload_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Replace NaN/NaT with empty strings for Google Sheets
        upload_df = upload_df.fillna('')
        
        # Upload to Google Sheets
        if GOOGLE_SHEETS_AVAILABLE:
            try:
                client = connect_to_google_sheets()
                spreadsheet = get_sheet(client)
                
                # Get or create worksheet
                try:
                    worksheet = spreadsheet.worksheet('reddit_data')
                except:
                    worksheet = spreadsheet.add_worksheet(title='reddit_data', rows=1000, cols=20)
                
                # Convert DataFrame to list of lists (all values as strings for safety)
                values = upload_df.astype(str).values.tolist()
                
                # Append data
                worksheet.append_rows(values, value_input_option='USER_ENTERED')
                
                print(f"   ✅ Uploaded to Google Sheets: {len(upload_df)} rows")
            except Exception as e:
                print(f"   ⚠️ Google Sheets error: {e}")
        
        # Upload to Supabase
        if SUPABASE_AVAILABLE:
            try:
                db = SupabaseDB()
                if db.is_connected():
                    success = db.insert_reddit_data(upload_df)
                    if success:
                        print(f"   ✅ Uploaded to Supabase: {len(upload_df)} rows")
                    else:
                        print(f"   ⚠️ Supabase upload failed")
                else:
                    print(f"   ⚠️ Supabase not connected")
            except Exception as e:
                print(f"   ⚠️ Supabase error: {e}")


# ==================== STANDALONE TESTING ====================

if __name__ == '__main__':
    """Test the collector with a sample topic"""
    
    print("🧪 Testing Dynamic Topic Collector\n")
    
    # Example topics to test
    test_topics = [
        "Nike running shoes",
        "iPhone 15 Pro",
        "Starbucks coffee"
    ]
    
    collector = DynamicTopicCollector()
    
    # Test with first topic
    topic = test_topics[0]
    print(f"\n📌 Testing with topic: '{topic}'")
    
    data = collector.collect_for_topic(topic, reddit_limit=50, youtube_limit=20)
    
    summary = collector.get_topic_summary(data)
    
    print("\n📊 Collection Summary:")
    print(f"Topic: {summary['topic']}")
    print(f"Total posts collected: {summary['total_posts']}")
    print(f"Reddit posts: {summary['reddit_posts']}")
    print(f"YouTube videos: {summary['youtube_videos']}")
    
    if 'reddit_insights' in summary:
        print(f"\nReddit Insights:")
        print(f"  - Average score: {summary['reddit_insights']['avg_score']:.1f}")
        print(f"  - Total comments: {summary['reddit_insights']['total_comments']}")
        print(f"  - Top subreddits: {list(summary['reddit_insights']['top_subreddits'].keys())}")
