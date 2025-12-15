"""
Backend utility module for Streamlit app
Handles all backend connections with proper caching and Supabase integration
"""

import streamlit as st
from engagement_predictor import EngagementPredictor
from campaign_simulator import CampaignSimulator
from prediction_coach import PredictionCoach
from google_sheet_connect import connect_to_google_sheets, get_sheet
import pandas as pd
from datetime import datetime, timedelta
import os

# Environment-based data source configuration
DATA_SOURCE = os.getenv("DATA_SOURCE", "SUPABASE")  # Default to Supabase for production

# Import Supabase integration
try:
    from utils.supabase_db import get_supabase_client
    SUPABASE_ENABLED = True
except ImportError:
    SUPABASE_ENABLED = False
    get_supabase_client = None
    print("⚠️ Supabase integration not available, using Google Sheets only")

# Log data source configuration
if DATA_SOURCE == "SUPABASE" and SUPABASE_ENABLED:
    print("🚀 Production mode: Supabase PRIMARY, Google Sheets FALLBACK")
elif DATA_SOURCE == "SHEETS":
    print("🔧 Development mode: Google Sheets PRIMARY")
else:
    print("⚠️ Fallback mode: Google Sheets ONLY (Supabase not configured)")



@st.cache_resource
def get_predictor():
    """Initialize and cache EngagementPredictor"""
    predictor = EngagementPredictor()
    predictor.load_model()
    return predictor


@st.cache_resource
def get_simulator():
    """Initialize and cache CampaignSimulator"""
    return CampaignSimulator()


@st.cache_resource
def get_coach():
    """Initialize and cache PredictionCoach"""
    return PredictionCoach()


@st.cache_resource
def get_google_sheets_connection():
    """Initialize and cache Google Sheets connection"""
    gc = connect_to_google_sheets()
    spreadsheet = get_sheet(gc)
    return gc, spreadsheet


@st.cache_data(ttl=0, show_spinner="Loading Twitter data...")
def load_twitter_data(days=None):
    """
    Twitter data loader
    PRIMARY: Supabase
    FALLBACK: Google Sheets (only if Supabase unavailable)
    """
    
    # ---------- TRY SUPABASE ----------
    if SUPABASE_ENABLED and get_supabase_client:
        supabase = get_supabase_client()
        if supabase.is_connected():
            df = supabase.get_twitter_data(days=days)
            
            if not df.empty:
                # Only format, do NOT filter (Supabase already filtered)
                df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
                
                df['likes'] = pd.to_numeric(df.get('likes', 0), errors='coerce').fillna(0)
                df['retweets'] = pd.to_numeric(df.get('retweets', 0), errors='coerce').fillna(0)
                df['replies'] = pd.to_numeric(df.get('replies', 0), errors='coerce').fillna(0)
                
                df['engagement'] = df['likes'] + df['retweets'] + df['replies']
                df.attrs['source'] = 'Supabase'
                
                print(f"🚀 [SUPABASE] Twitter rows (FINAL): {len(df)}")
                return df
            
            st.warning("⚠️ Supabase connected but returned 0 Twitter rows")
    
    # ---------- FALLBACK: GOOGLE SHEETS ----------
    st.warning("📄 Falling back to Google Sheets (Twitter)")
    try:
        _, spreadsheet = get_google_sheets_connection()
        sheet = spreadsheet.worksheet("twitter_data")
        data = sheet.get_all_values()
        
        if len(data) > 1:
            df = pd.DataFrame(data[1:], columns=data[0])
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            df['likes'] = pd.to_numeric(df['likes'], errors='coerce').fillna(0)
            df['retweets'] = pd.to_numeric(df['retweets'], errors='coerce').fillna(0)
            df['replies'] = pd.to_numeric(df['replies'], errors='coerce').fillna(0)
            df['engagement'] = df['likes'] + df['retweets'] + df['replies']
            df.attrs['source'] = 'Google Sheets'
            
            print(f"📊 [SHEETS] Twitter rows: {len(df)}")
            return df
    except Exception as e:
        st.error(f"❌ Error loading Twitter data: {e}")
    
    return pd.DataFrame()


@st.cache_data(ttl=0, show_spinner="Loading Reddit data...")
def load_reddit_data(days=None):
    """
    Reddit data loader
    PRIMARY: Supabase
    FALLBACK: Google Sheets
    """
    
    # ---------- TRY SUPABASE ----------
    if SUPABASE_ENABLED and get_supabase_client:
        supabase = get_supabase_client()
        if supabase.is_connected():
            df = supabase.get_reddit_data(days=days)
            
            if not df.empty:
                # Only format, do NOT filter (Supabase already filtered)
                df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
                
                df['score'] = pd.to_numeric(df.get('score', 0), errors='coerce').fillna(0)
                df['num_comments'] = pd.to_numeric(df.get('num_comments', 0), errors='coerce').fillna(0)
                
                df['engagement'] = df['score'] + df['num_comments']
                df.attrs['source'] = 'Supabase'
                
                print(f"🚀 [SUPABASE] Reddit rows (FINAL): {len(df)}")
                return df
            
            st.warning("⚠️ Supabase connected but returned 0 Reddit rows")
    
    # ---------- FALLBACK ----------
    st.warning("📄 Falling back to Google Sheets (Reddit)")
    try:
        _, spreadsheet = get_google_sheets_connection()
        sheet = spreadsheet.worksheet("reddit_data")
        data = sheet.get_all_values()
        
        if len(data) > 1:
            df = pd.DataFrame(data[1:], columns=data[0])
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0)
            df['num_comments'] = pd.to_numeric(df['num_comments'], errors='coerce').fillna(0)
            df['engagement'] = df['score'] + df['num_comments']
            df.attrs['source'] = 'Google Sheets'
            
            print(f"📊 [SHEETS] Reddit rows: {len(df)}")
            return df
    except Exception as e:
        st.error(f"❌ Error loading Reddit data: {e}")
    
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_sentiment_data():
    """Load sentiment analysis data from Google Sheets"""
    try:
        _, spreadsheet = get_google_sheets_connection()
        sheet = spreadsheet.worksheet('Sentiment_Insights')
        data = sheet.get_all_values()
        
        if len(data) > 1:
            df = pd.DataFrame(data[1:], columns=data[0])
            return df
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()


def write_to_google_sheets(sheet_name, data_rows, headers=None):
    """Write data to Google Sheets"""
    try:
        _, spreadsheet = get_google_sheets_connection()
        
        # Get or create worksheet
        try:
            worksheet = spreadsheet.worksheet(sheet_name)
        except:
            worksheet = spreadsheet.add_worksheet(
                title=sheet_name,
                rows=1000,
                cols=20
            )
            if headers:
                worksheet.update(range_name='A1', values=[headers])
        
        # Append data
        worksheet.append_rows(data_rows, value_input_option='USER_ENTERED')
        return True
    except Exception as e:
        st.error(f"Error writing to Google Sheets: {e}")
        return False


def get_last_30_days_data(df, date_column='created_at'):
    """Filter dataframe to last 30 days"""
    if len(df) == 0:
        return df
    
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Make timezone-aware comparison
    thirty_days_ago = datetime.now() - timedelta(days=30)
    
    # Ensure the date column is datetime and handle timezone
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    # Remove timezone info from both for comparison
    if df[date_column].dt.tz is not None:
        df[date_column] = df[date_column].dt.tz_localize(None)
    
    df_filtered = df[df[date_column] >= thirty_days_ago].copy()
    
    # Ensure engagement column exists and is numeric
    if 'engagement' not in df_filtered.columns:
        if 'likes' in df_filtered.columns:
            df_filtered['engagement'] = pd.to_numeric(df_filtered.get('likes', 0), errors='coerce').fillna(0) + \
                                        pd.to_numeric(df_filtered.get('retweets', 0), errors='coerce').fillna(0) + \
                                        pd.to_numeric(df_filtered.get('replies', 0), errors='coerce').fillna(0)
        elif 'score' in df_filtered.columns:
            df_filtered['engagement'] = pd.to_numeric(df_filtered.get('score', 0), errors='coerce').fillna(0) + \
                                        pd.to_numeric(df_filtered.get('num_comments', 0), errors='coerce').fillna(0)
    else:
        # Ensure existing engagement column is numeric
        df_filtered['engagement'] = pd.to_numeric(df_filtered['engagement'], errors='coerce').fillna(0)
    
    return df_filtered


def send_slack_notification(message):
    """Send notification to Slack"""
    try:
        from slack_notify import send_slack_message
        send_slack_message(message, emoji=":robot_face:")
        return True
    except Exception as e:
        st.error(f"Error sending Slack notification: {e}")
        return False
