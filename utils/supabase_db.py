"""
Supabase Database Integration
Provides fast queries and unlimited concurrent access for production deployment
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("⚠️ Supabase not installed. Run: pip install supabase")


class SupabaseDB:
    """Supabase database wrapper for social media data"""
    
    def __init__(self):
        self.client = None
        self.connected = False
        self._connect()
    
    def _connect(self):
        """Connect to Supabase using secrets"""
        if not SUPABASE_AVAILABLE:
            return False
        
        try:
            # Try to get credentials from Streamlit secrets
            if hasattr(st, 'secrets') and 'supabase' in st.secrets:
                url = st.secrets['supabase']['url']
                key = st.secrets['supabase']['key']
                
                # Create client - simple positional arguments
                # This is the correct way for supabase-py 2.x
                try:
                    self.client = create_client(url, key)
                    self.connected = True
                    return True
                except TypeError as te:
                    # If there's a TypeError, it might be a version issue
                    print(f"⚠️ Supabase client creation failed: {te}")
                    print(f"⚠️ Try: pip install --upgrade supabase")
                    return False
            else:
                print("⚠️ Supabase credentials not found in secrets")
                return False
                
        except Exception as e:
            print(f"⚠️ Supabase connection error: {e}")
            self.connected = False
            return False
    
    def is_connected(self):
        """Check if connected to Supabase"""
        return self.connected and self.client is not None
    
    # ==================== TWITTER DATA ====================
    
    def get_twitter_data(self, days=None):
        """Get Twitter data from last N days (None = all time)
        
        Uses pagination to fetch ALL rows, bypassing Supabase's default/max limits.
        
        Args:
            days: Number of days to fetch (None = all time, NO DEFAULT)
        """
        if not self.is_connected():
            return pd.DataFrame()
        
        try:
            all_data = []
            page_size = 1000  # Fetch in batches
            offset = 0
            
            # Apply date filter if specified
            if days is not None:
                cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
                print(f"🔍 [SUPABASE] Twitter filter: last {days} days (cutoff: {cutoff_date})")
            else:
                print(f"🔍 [SUPABASE] Twitter filter: ALL TIME (no date filter)")
                cutoff_date = None
            
            # Paginate through all results
            while True:
                query = self.client.table('twitter_data').select('*')
                
                # Apply date filter
                if cutoff_date:
                    query = query.gte('created_at', cutoff_date)
                
                # Apply pagination
                query = query.order('created_at', desc=True).range(offset, offset + page_size - 1)
                
                response = query.execute()
                
                if not response.data or len(response.data) == 0:
                    break  # No more data
                
                all_data.extend(response.data)
                print(f"✅ [SUPABASE] Fetched batch: {len(response.data)} rows (total so far: {len(all_data)})")
                
                # If we got less than page_size, we've reached the end
                if len(response.data) < page_size:
                    break
                
                offset += page_size
            
            if all_data:
                df = pd.DataFrame(all_data)
                print(f"🚀 [SUPABASE] Twitter rows (FINAL): {len(df)}")
                return df
            return pd.DataFrame()
            
        except Exception as e:
            print(f"❌ [SUPABASE] Error fetching Twitter data: {e}")
            return pd.DataFrame()
    
    def insert_twitter_data(self, df):
        """Insert Twitter data handling mixed date formats (ISO and MM/DD/YYYY)"""
        if not self.is_connected() or df.empty:
            return False
        
        try:
            df = df.copy()
            
            # --- CRITICAL FIX: Handle Mixed Twitter Date Formats ---
            # Handles both '2025-10-29...' and '10/27/2025...' automatically
            if 'created_at' in df.columns:
                # 1. Force convert to datetime using 'mixed' format inference
                df['created_at'] = pd.to_datetime(df['created_at'], format='mixed', errors='coerce', utc=True)
                
                # 2. Convert to ISO 8601 strings for Supabase/Postgres
                # NaT (Not a Time) values are converted to None
                df['created_at'] = df['created_at'].apply(lambda x: x.isoformat() if pd.notnull(x) else None)

            # --- Safe Numeric Fill ---
            # Fill NaN with 0 for numbers, but ignore ID columns
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            numeric_cols = [c for c in numeric_cols if 'id' not in c.lower()]
            df[numeric_cols] = df[numeric_cols].fillna(0)
            
            # --- Dictionary Cleanup ---
            records = df.to_dict('records')
            cleaned_records = []
            
            for record in records:
                # Filter out None/NaN values so database uses NULL
                clean_rec = {k: v for k, v in record.items() if v is not None}
                
                # Double check: if created_at is None, remove the key entirely
                if 'created_at' in clean_rec and clean_rec['created_at'] is None:
                    del clean_rec['created_at']
                    
                cleaned_records.append(clean_rec)

            # Insert in batches
            batch_size = 1000
            for i in range(0, len(cleaned_records), batch_size):
                batch = cleaned_records[i:i + batch_size]
                if batch:
                    self.client.table('twitter_data').upsert(batch).execute()
            
            print(f"✅ Inserted {len(cleaned_records)} Twitter records")
            return True
            
        except Exception as e:
            print(f"❌ Error inserting Twitter data: {e}")
            return False
    
    # ==================== REDDIT DATA ====================
    
    def get_reddit_data(self, days=None):
        """Get Reddit data from last N days (None = all time)
        
        Uses pagination to fetch ALL rows, bypassing Supabase's default/max limits.
        
        Args:
            days: Number of days to fetch (None = all time, NO DEFAULT)
        """
        if not self.is_connected():
            return pd.DataFrame()
        
        try:
            all_data = []
            page_size = 1000  # Fetch in batches
            offset = 0
            
            # Apply date filter if specified
            if days is not None:
                cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
                print(f"🔍 [SUPABASE] Reddit filter: last {days} days (cutoff: {cutoff_date})")
            else:
                print(f"🔍 [SUPABASE] Reddit filter: ALL TIME (no date filter)")
                cutoff_date = None
            
            # Paginate through all results
            while True:
                query = self.client.table('reddit_data').select('*')
                
                # Apply date filter
                if cutoff_date:
                    query = query.gte('created_at', cutoff_date)
                
                # Apply pagination
                query = query.order('created_at', desc=True).range(offset, offset + page_size - 1)
                
                response = query.execute()
                
                if not response.data or len(response.data) == 0:
                    break  # No more data
                
                all_data.extend(response.data)
                print(f"✅ [SUPABASE] Fetched batch: {len(response.data)} rows (total so far: {len(all_data)})")
                
                # If we got less than page_size, we've reached the end
                if len(response.data) < page_size:
                    break
                
                offset += page_size
            
            if all_data:
                df = pd.DataFrame(all_data)
                print(f"🚀 [SUPABASE] Reddit rows (FINAL): {len(df)}")
                return df
            return pd.DataFrame()
            
        except Exception as e:
            print(f"❌ [SUPABASE] Error fetching Reddit data: {e}")
            return pd.DataFrame()
    
    def insert_reddit_data(self, df):
        """Insert Reddit data with dynamic topic collection support - EXACT SCHEMA MATCH"""
        if not self.is_connected() or df.empty:
            return False
        
        try:
            df = df.copy()
            
            # Define EXACT Supabase table schema - NO 'text', ONLY 'selftext'
            schema_columns = [
                'post_id', 'title', 'clean_title', 'author', 'subreddit', 
                'created_at', 'score', 'num_comments', 'upvote_ratio', 'url',
                'selftext', 'permalink', 'is_self', 'topic', 'search_query', 
                'platform', 'collected_at'
            ]
            
            # Handle legacy 'text' field from old code - convert to 'selftext'
            if 'text' in df.columns:
                if 'selftext' not in df.columns:
                    df['selftext'] = df['text']
                df = df.drop(columns=['text'])
                print(f"   ⚠️ Converted 'text' to 'selftext' (legacy compatibility)")
            
            # Remove created_utc if present (legacy field)
            if 'created_utc' in df.columns:
                df = df.drop(columns=['created_utc'])
                print(f"   ⚠️ Dropped 'created_utc' (use 'created_at' only)")
            
            # Ensure clean_title exists
            if 'clean_title' not in df.columns and 'title' in df.columns:
                df['clean_title'] = df['title']
            
            # Keep only schema columns
            cols_to_keep = [col for col in schema_columns if col in df.columns]
            df = df[cols_to_keep]
            
            # Verify no extra columns remain
            extra_cols = [col for col in df.columns if col not in schema_columns]
            if extra_cols:
                print(f"   ❌ ERROR: Extra columns found: {extra_cols}")
                df = df.drop(columns=extra_cols)

            # --- DATE HANDLING: Convert all timestamp columns to ISO format ---
            # Handles: created_at, collected_at (both can be strings or datetime objects)
            # Output: ISO 8601 string with timezone for PostgreSQL TIMESTAMPTZ
            
            timestamp_columns = ['created_at', 'collected_at']
            for col in timestamp_columns:
                if col in df.columns:
                    # Convert using mixed format to handle any date variations
                    df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
                    
                    # Convert to ISO 8601 string for PostgreSQL TIMESTAMPTZ
                    df[col] = df[col].apply(lambda x: x.isoformat() if pd.notnull(x) else None)

            # --- Safe Numeric Fill ---
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            numeric_cols = [c for c in numeric_cols if 'id' not in c.lower()]
            df[numeric_cols] = df[numeric_cols].fillna(0)

            records = df.to_dict('records')

            # --- Dictionary Cleanup ---
            cleaned_records = []
            for record in records:
                # Only keep non-None values that are in schema
                clean_rec = {k: v for k, v in record.items() 
                           if v is not None and k in schema_columns}
                
                # Ensure timestamp fields are not None (remove if None)
                for ts_col in timestamp_columns:
                    if ts_col in clean_rec and clean_rec[ts_col] is None:
                        del clean_rec[ts_col]
                
                # Final validation: no 'text' or 'created_utc' fields
                if 'text' in clean_rec:
                    del clean_rec['text']
                if 'created_utc' in clean_rec:
                    del clean_rec['created_utc']
                
                cleaned_records.append(clean_rec)

            # Insert in batches
            batch_size = 100  # Smaller batches for better error handling
            total_inserted = 0
            
            for i in range(0, len(cleaned_records), batch_size):
                batch = cleaned_records[i:i + batch_size]
                if not batch:
                    continue
                    
                try:
                    # Try upsert first
                    response = self.client.table('reddit_data').upsert(batch).execute()
                    total_inserted += len(batch)
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"   ⚠️ Batch upsert failed: {error_msg[:100]}")
                    
                    # If schema cache error (PGRST204), insert records one by one
                    if 'PGRST204' in error_msg or 'created_utc' in error_msg:
                        print(f"   🔄 Schema cache issue detected, inserting one-by-one...")
                        
                        for record in batch:
                            try:
                                # Try insert first
                                self.client.table('reddit_data').insert(record).execute()
                                total_inserted += 1
                            except Exception as insert_error:
                                # If duplicate key error, try update
                                if 'duplicate' in str(insert_error).lower() or '23505' in str(insert_error):
                                    try:
                                        post_id = record.get('post_id')
                                        if post_id:
                                            self.client.table('reddit_data')\
                                                .update(record)\
                                                .eq('post_id', post_id)\
                                                .execute()
                                            total_inserted += 1
                                    except Exception as update_error:
                                        print(f"   ❌ Failed to update {post_id}: {update_error}")
                                else:
                                    print(f"   ❌ Failed to insert record: {insert_error}")
                    else:
                        # Different error, re-raise
                        raise e
            
            print(f"✅ Inserted {total_inserted} Reddit records")
            return True
            
        except Exception as e:
            print(f"❌ Error inserting Reddit data: {e}")
            return False
    
    # ==================== A/B TEST RESULTS ====================
    
    def get_ab_tests(self, limit=50):
        """Get A/B test results"""
        if not self.is_connected():
            return pd.DataFrame()
        
        try:
            response = self.client.table('ab_test_results')\
                .select('*')\
                .order('timestamp', desc=True)\
                .limit(limit)\
                .execute()
            
            if response.data:
                return pd.DataFrame(response.data)
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching A/B tests: {e}")
            return pd.DataFrame()
    
    def insert_ab_test(self, test_data):
        """Insert A/B test result"""
        if not self.is_connected():
            return False
        
        try:
            self.client.table('ab_test_results').insert(test_data).execute()
            return True
        except Exception as e:
            print(f"Error inserting A/B test: {e}")
            return False
    
    # ==================== STATS ====================
    
    def get_stats(self):
        """Get database statistics"""
        if not self.is_connected():
            return {}
        
        try:
            stats = {
                'twitter_count': 0,
                'reddit_count': 0,
                'ab_tests_count': 0,
                'last_updated': None
            }
            
            # Count Twitter records
            twitter_response = self.client.table('twitter_data')\
                .select('*', count='exact')\
                .execute()
            stats['twitter_count'] = twitter_response.count if hasattr(twitter_response, 'count') else 0
            
            # Count Reddit records
            reddit_response = self.client.table('reddit_data')\
                .select('*', count='exact')\
                .execute()
            stats['reddit_count'] = reddit_response.count if hasattr(reddit_response, 'count') else 0
            
            # Count A/B tests
            ab_response = self.client.table('ab_test_results')\
                .select('*', count='exact')\
                .execute()
            stats['ab_tests_count'] = ab_response.count if hasattr(ab_response, 'count') else 0
            
            # Get last updated timestamp
            latest_twitter = self.client.table('twitter_data')\
                .select('created_at')\
                .order('created_at', desc=True)\
                .limit(1)\
                .execute()
            
            if latest_twitter.data:
                stats['last_updated'] = latest_twitter.data[0]['created_at']
            
            return stats
            
        except Exception as e:
            print(f"Error fetching stats: {e}")
            return {}


# ==================== CACHED INSTANCE ====================

@st.cache_resource
def get_supabase_client():
    """Get cached Supabase client instance"""
    return SupabaseDB()
