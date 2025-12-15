"""
Data Sync Service
Syncs data from Google Sheets to Supabase for fast queries
"""

import pandas as pd
from datetime import datetime
from google_sheet_connect import connect_to_google_sheets, get_sheet
from utils.supabase_db import get_supabase_client
import streamlit as st


def sync_google_sheets_to_supabase():
    """
    Sync all data from Google Sheets to Supabase
    Returns: (success, message)
    """
    
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        
        if not supabase.is_connected():
            return False, "Supabase not connected. Using Google Sheets only."
        
        # Connect to Google Sheets
        gc = connect_to_google_sheets()
        if not gc:
            return False, "Failed to connect to Google Sheets"
        
        spreadsheet = get_sheet(gc)
        if not spreadsheet:
            return False, "Failed to open spreadsheet"
        
        sync_results = {
            'twitter': 0,
            'reddit': 0,
            'errors': []
        }
        
        # ==================== SYNC TWITTER DATA ====================
        try:
            twitter_worksheet = spreadsheet.worksheet('Twitter Data')
            # Use UNFORMATTED_VALUE to preserve large tweet_id numbers (avoid scientific notation like 1.99869E+18)
            twitter_data = twitter_worksheet.get_all_values(value_render_option='UNFORMATTED_VALUE')
            
            if len(twitter_data) > 1:
                # Handle empty headers
                headers = [h if h else f'column_{i}' for i, h in enumerate(twitter_data[0])]
                twitter_df = pd.DataFrame(twitter_data[1:], columns=headers)
                
                print(f"📊 Loaded {len(twitter_df)} Twitter rows from Google Sheets")
                
                # Convert tweet_id to string immediately to preserve full number
                if 'tweet_id' in twitter_df.columns:
                    twitter_df['tweet_id'] = twitter_df['tweet_id'].astype(str)
                    print(f"📊 Sample tweet_ids: {twitter_df['tweet_id'].head(3).tolist()}")
                
                # Clean and prepare created_at datetime column
                # Twitter has TWO formats: "2025-10-29 09:40:31+00:00" OR "10/27/2025 23:31:00"
                if 'created_at' in twitter_df.columns:
                    # Replace invalid values first
                    twitter_df['created_at'] = twitter_df['created_at'].astype(str).replace(['', '0', '0.0', 'null', 'NULL', 'None', 'nan'], None)
                    
                    # Strip timezone info (e.g., "+00:00") if present
                    twitter_df['created_at'] = twitter_df['created_at'].astype(str).str.replace(r'[+\-]\d{2}:\d{2}$', '', regex=True)
                    
                    # Parse dates - pandas will auto-detect both formats with errors='coerce'
                    twitter_df['created_at'] = pd.to_datetime(twitter_df['created_at'], errors='coerce')
                    
                    # Replace NaT with None
                    twitter_df['created_at'] = twitter_df['created_at'].where(pd.notna(twitter_df['created_at']), None)
                    
                    # Convert to string format for database
                    twitter_df['created_at'] = twitter_df['created_at'].apply(
                        lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if x is not None and pd.notna(x) else None
                    )
                    
                    print(f"📊 Sample Twitter created_at: {twitter_df['created_at'].head(3).tolist()}")
                
                # Convert boolean column
                if 'verified' in twitter_df.columns:
                    twitter_df['verified'] = twitter_df['verified'].map({'TRUE': True, 'FALSE': False, True: True, False: False})
                    twitter_df['verified'] = twitter_df['verified'].fillna(False).infer_objects(copy=False).astype(bool)
                
                # Convert numeric columns (use infer_objects to avoid FutureWarning)
                if 'likes' in twitter_df.columns:
                    twitter_df['likes'] = pd.to_numeric(twitter_df['likes'], errors='coerce').fillna(0).infer_objects(copy=False).astype(int)
                if 'retweets' in twitter_df.columns:
                    twitter_df['retweets'] = pd.to_numeric(twitter_df['retweets'], errors='coerce').fillna(0).infer_objects(copy=False).astype(int)
                if 'replies' in twitter_df.columns:
                    twitter_df['replies'] = pd.to_numeric(twitter_df['replies'], errors='coerce').fillna(0).infer_objects(copy=False).astype(int)
                
                # Calculate engagement (will be auto-generated in DB, but needed for display)
                if 'engagement' not in twitter_df.columns and all(col in twitter_df.columns for col in ['likes', 'retweets', 'replies']):
                    twitter_df['engagement'] = twitter_df['likes'] + twitter_df['retweets'] + twitter_df['replies']
                
                # Replace all NaN and inf values in numeric columns only
                numeric_columns = twitter_df.select_dtypes(include=['float64', 'int64']).columns
                twitter_df[numeric_columns] = twitter_df[numeric_columns].fillna(0).infer_objects(copy=False)
                twitter_df[numeric_columns] = twitter_df[numeric_columns].replace([float('inf'), float('-inf')], 0)
                
                # Remove duplicate tweet_ids (keep first occurrence)
                if 'tweet_id' in twitter_df.columns:
                    twitter_df = twitter_df.drop_duplicates(subset=['tweet_id'], keep='first')
                
                # Select columns matching new Supabase schema (REMOVED fetched_at - not important)
                # Schema: tweet_id (PK), created_at, text, cleaned_text, author_id, author_username, author_name, verified, likes, retweets, replies, language, source, engagement (generated)
                supabase_columns = ['tweet_id', 'created_at', 'text', 'cleaned_text', 
                                   'author_id', 'author_username', 'author_name', 'verified',
                                   'likes', 'retweets', 'replies', 'language', 'source']
                
                # Filter to only include columns that exist in both
                available_columns = [col for col in supabase_columns if col in twitter_df.columns]
                twitter_df = twitter_df[available_columns]
                
                # Final cleanup: replace any remaining "0" strings in created_at with None
                if 'created_at' in twitter_df.columns:
                    twitter_df['created_at'] = twitter_df['created_at'].replace('0', None)
                    # Debug: Check for problematic values
                    problematic = twitter_df[twitter_df['created_at'] == '0']
                    if len(problematic) > 0:
                        print(f"   ⚠️ Found {len(problematic)} tweets with '0' in created_at - replacing with None")
                    
                    # Check data types
                    print(f"   📊 Twitter created_at column type: {twitter_df['created_at'].dtype}")
                    print(f"   📊 Sample created_at values: {twitter_df['created_at'].head(3).tolist()}")
                
                # Insert into Supabase
                if supabase.insert_twitter_data(twitter_df):
                    sync_results['twitter'] = len(twitter_df)
                else:
                    sync_results['errors'].append("Failed to sync Twitter data")
            
        except Exception as e:
            sync_results['errors'].append(f"Twitter sync error: {str(e)}")
        
        # ==================== SYNC REDDIT DATA ====================
        try:
            reddit_worksheet = spreadsheet.worksheet('Reddit Data')
            # Use UNFORMATTED_VALUE to preserve post_id strings
            reddit_data = reddit_worksheet.get_all_values(value_render_option='UNFORMATTED_VALUE')
            
            if len(reddit_data) > 1:
                # Filter out empty headers and handle duplicates
                headers = [h if h else f'column_{i}' for i, h in enumerate(reddit_data[0])]
                reddit_df = pd.DataFrame(reddit_data[1:], columns=headers)
                
                print(f"📊 Loaded {len(reddit_df)} Reddit rows from Google Sheets")
                
                # Convert post_id to string to preserve full ID
                if 'post_id' in reddit_df.columns:
                    reddit_df['post_id'] = reddit_df['post_id'].astype(str)
                    print(f"📊 Sample post_ids: {reddit_df['post_id'].head(3).tolist()}")
                
                # Clean and prepare datetime (Reddit has ISO format: "2025-10-27 23:55:51")
                if 'created_at' in reddit_df.columns:
                    # Replace invalid values first
                    reddit_df['created_at'] = reddit_df['created_at'].astype(str).replace(['', '0', '0.0', 'null', 'NULL', 'None', 'nan'], None)
                    
                    # Parse Reddit date - already in ISO format YYYY-MM-DD HH:MM:SS
                    reddit_df['created_at'] = pd.to_datetime(
                        reddit_df['created_at'], 
                        format='%Y-%m-%d %H:%M:%S',  # ISO format
                        errors='coerce'
                    )
                    
                    # Replace NaT with None
                    reddit_df['created_at'] = reddit_df['created_at'].where(pd.notna(reddit_df['created_at']), None)
                    
                    # Convert to string format for database
                    reddit_df['created_at'] = reddit_df['created_at'].apply(
                        lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if x is not None and pd.notna(x) else None
                    )
                    
                    print(f"📊 Sample Reddit created_at: {reddit_df['created_at'].head(3).tolist()}")
                
                # Convert numeric columns (use infer_objects to avoid FutureWarning)
                if 'score' in reddit_df.columns:
                    reddit_df['score'] = pd.to_numeric(reddit_df['score'], errors='coerce').fillna(0).infer_objects(copy=False).astype(int)
                if 'num_comments' in reddit_df.columns:
                    reddit_df['num_comments'] = pd.to_numeric(reddit_df['num_comments'], errors='coerce').fillna(0).infer_objects(copy=False).astype(int)
                if 'upvote_ratio' in reddit_df.columns:
                    reddit_df['upvote_ratio'] = pd.to_numeric(reddit_df['upvote_ratio'], errors='coerce').fillna(0).infer_objects(copy=False)
                
                # Calculate engagement (will be auto-generated in DB, but needed for display)
                if 'engagement' not in reddit_df.columns and all(col in reddit_df.columns for col in ['score', 'num_comments']):
                    reddit_df['engagement'] = reddit_df['score'] + reddit_df['num_comments']
                
                # Replace all NaN and inf values in numeric columns only
                numeric_columns = reddit_df.select_dtypes(include=['float64', 'int64']).columns
                reddit_df[numeric_columns] = reddit_df[numeric_columns].fillna(0).infer_objects(copy=False)
                reddit_df[numeric_columns] = reddit_df[numeric_columns].replace([float('inf'), float('-inf')], 0)
                
                # Remove duplicate post_ids (keep first occurrence) - CRITICAL for Reddit
                if 'post_id' in reddit_df.columns:
                    initial_count = len(reddit_df)
                    reddit_df = reddit_df.drop_duplicates(subset=['post_id'], keep='first')
                    removed_count = initial_count - len(reddit_df)
                    if removed_count > 0:
                        print(f"   ⚠️ Removed {removed_count} duplicate Reddit post_ids")
                
                # Select columns matching new Supabase schema
                # Schema: post_id (PK), title, clean_title, author, subreddit, created_at, score, num_comments, upvote_ratio, url, engagement (generated)
                supabase_columns = ['post_id', 'title', 'clean_title', 'author', 'subreddit', 
                                   'created_at', 'score', 'num_comments', 'upvote_ratio', 'url']
                
                # Filter to only include columns that exist in both
                available_columns = [col for col in supabase_columns if col in reddit_df.columns]
                
                print(f"   📋 Columns before filter: {list(reddit_df.columns)}")
                print(f"   📋 Columns to keep: {available_columns}")
                
                # Select ONLY the columns we want (this removes created_utc and others)
                reddit_df = reddit_df[available_columns].copy()
                
                print(f"   📋 Columns after filter: {list(reddit_df.columns)}")
                
                # Insert into Supabase
                if supabase.insert_reddit_data(reddit_df):
                    sync_results['reddit'] = len(reddit_df)
                else:
                    sync_results['errors'].append("Failed to sync Reddit data")
            
        except Exception as e:
            sync_results['errors'].append(f"Reddit sync error: {str(e)}")
        
        # Build result message
        if sync_results['twitter'] > 0 or sync_results['reddit'] > 0:
            message = f"✅ Synced {sync_results['twitter']} Twitter posts and {sync_results['reddit']} Reddit posts"
            if sync_results['errors']:
                message += f"\n⚠️ Warnings: {', '.join(sync_results['errors'])}"
            return True, message
        else:
            if sync_results['errors']:
                return False, f"❌ Sync failed: {', '.join(sync_results['errors'])}"
            else:
                return True, "No new data to sync"
        
    except Exception as e:
        return False, f"❌ Sync error: {str(e)}"


def get_last_sync_time():
    """Get the last time data was synced"""
    try:
        supabase = get_supabase_client()
        if supabase.is_connected():
            stats = supabase.get_stats()
            return stats.get('last_updated', 'Never')
    except:
        pass
    return 'Never'


@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_data_with_fallback(source='twitter', days=30):
    """
    Load data with fallback: Try Supabase first, then Google Sheets
    
    Args:
        source: 'twitter' or 'reddit'
        days: Number of days to load (None = all time)
    
    Returns: DataFrame
    """
    
    # Try Supabase first (FAST)
    try:
        supabase = get_supabase_client()
        
        if supabase.is_connected():
            if source == 'twitter':
                df = supabase.get_twitter_data(days=days if days else None)
                if not df.empty:
                    return df
            elif source == 'reddit':
                df = supabase.get_reddit_data(days=days if days else None)
                if not df.empty:
                    return df
    except Exception as e:
        print(f"⚠️ Supabase fetch failed, falling back to Google Sheets: {e}")
    
    # Fallback to Google Sheets (SLOW but reliable)
    try:
        gc = connect_to_google_sheets()
        if gc:
            spreadsheet = get_sheet(gc)
            if spreadsheet:
                worksheet_name = 'twitter_data' if source == 'twitter' else 'reddit_data'
                worksheet = spreadsheet.worksheet(worksheet_name)
                data = worksheet.get_all_values()
                
                if len(data) > 1:
                    # Handle empty headers
                    headers = [h if h else f'column_{i}' for i, h in enumerate(data[0])]
                    df = pd.DataFrame(data[1:], columns=headers)
                    
                    # Filter by date only if days is specified
                    if days is not None:
                        if source == 'twitter' and 'created_at' in df.columns:
                            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True)
                            # Remove timezone for comparison
                            if df['created_at'].dt.tz is not None:
                                df['created_at'] = df['created_at'].dt.tz_localize(None)
                            cutoff = datetime.now() - pd.Timedelta(days=days)
                            df = df[df['created_at'] >= cutoff]
                        elif source == 'reddit' and 'created_at' in df.columns:
                            # Reddit now uses created_at (TIMESTAMPTZ) instead of created_utc
                            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
                            if df['created_at'].dt.tz is not None:
                                df['created_at'] = df['created_at'].dt.tz_localize(None)
                            cutoff = datetime.now() - pd.Timedelta(days=days)
                            df = df[df['created_at'] >= cutoff]
                    else:
                        # No date filtering for all-time data
                        # Still need to parse dates for proper sorting/display
                        if 'created_at' in df.columns:
                            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True)
                    
                    return df
    except Exception as e:
        print(f"❌ Google Sheets fallback failed: {e}")
    
    # Return empty DataFrame if both fail
    return pd.DataFrame()
