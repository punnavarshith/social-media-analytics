"""
API Configuration for Dynamic Topic Collection
Automatically loads credentials from Streamlit secrets or .env file
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env (local development)
load_dotenv()

# Try Streamlit secrets first (cloud deployment)
try:
    import streamlit as st
    REDDIT_CONFIG = {
        'client_id': st.secrets.get("reddit", {}).get("client_id") or os.getenv('REDDIT_CLIENT_ID'),
        'client_secret': st.secrets.get("reddit", {}).get("client_secret") or os.getenv('REDDIT_CLIENT_SECRET'),
        'user_agent': st.secrets.get("reddit", {}).get("user_agent") or os.getenv('REDDIT_USER_AGENT', 'DynamicTopicAnalyzer/1.0')
    }
    YOUTUBE_API_KEY = st.secrets.get('YOUTUBE_API_KEY', '') or os.getenv('YOUTUBE_API_KEY', '')
except:
    # Fallback to environment variables only (local)
    REDDIT_CONFIG = {
        'client_id': os.getenv('REDDIT_CLIENT_ID'),
        'client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
        'user_agent': os.getenv('REDDIT_USER_AGENT', 'DynamicTopicAnalyzer/1.0')
    }
    YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', '')

# ==================== YOUTUBE API (Optional) ====================
# Get from: https://console.cloud.google.com/
# Enable YouTube Data API v3 and create API key

YOUTUBE_ENABLED = bool(YOUTUBE_API_KEY)  # Automatically enabled if key exists
