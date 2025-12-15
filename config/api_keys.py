"""
API Configuration for Dynamic Topic Collection
Automatically loads credentials from .env file
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==================== REDDIT API ====================
# Uses existing credentials from .env file
# No need to configure - already set up!

REDDIT_CONFIG = {
    'client_id': os.getenv('REDDIT_CLIENT_ID'),
    'client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
    'user_agent': os.getenv('REDDIT_USER_AGENT', 'DynamicTopicAnalyzer/1.0')
}

# ==================== YOUTUBE API (Optional) ====================
# Get from: https://console.cloud.google.com/
# Enable YouTube Data API v3 and create API key

YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', '')  # Optional - add to .env if you have it
YOUTUBE_ENABLED = bool(YOUTUBE_API_KEY)  # Automatically enabled if key exists
