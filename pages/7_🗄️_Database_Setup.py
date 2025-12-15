"""
Database Setup & Sync Management Page
Configure Supabase and manage data synchronization
"""

import streamlit as st
from utils.styles import get_custom_css
from utils.supabase_db import get_supabase_client
from utils.data_sync import sync_google_sheets_to_supabase, get_last_sync_time
from datetime import datetime

st.set_page_config(page_title="Database Setup", page_icon="🗄️", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Header
st.markdown('<h1 class="page-title">🗄️ Database Setup & Sync</h1>', unsafe_allow_html=True)
st.markdown("Configure Supabase for production deployment and manage data synchronization")
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ==================== CONNECTION STATUS ====================
st.markdown('<h3 class="section-header">📊 Connection Status</h3>', unsafe_allow_html=True)

supabase = get_supabase_client()

col1, col2 = st.columns(2)

with col1:
    if supabase.is_connected():
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("""
        ### ✅ Supabase Connected
        
        **Status:** Active  
        **Mode:** Production  
        **Performance:** High-speed queries  
        **Concurrent Users:** Unlimited
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Get stats
        stats = supabase.get_stats()
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Twitter Records", f"{stats.get('twitter_count', 0):,}")
        with col_b:
            st.metric("Reddit Records", f"{stats.get('reddit_count', 0):,}")
        with col_c:
            st.metric("A/B Tests", f"{stats.get('ab_tests_count', 0):,}")
        
    else:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("""
        ### ⚠️ Supabase Not Connected
        
        **Status:** Using Google Sheets only  
        **Mode:** Development  
        **Limitation:** Rate limits may apply
        
        Configure Supabase credentials below for production deployment.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 📈 Why Supabase?
    
    **Benefits:**
    - ⚡ 100x faster queries
    - 🚀 Unlimited concurrent users
    - 🔒 No API rate limits
    - 💾 500 MB free storage
    - 🌍 Global CDN
    - 🆓 100% FREE forever
    
    **Perfect for public deployment!**
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ==================== SETUP INSTRUCTIONS ====================
if not supabase.is_connected():
    st.markdown('<h3 class="section-header">🚀 Setup Instructions</h3>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### Step 1: Create Supabase Account
    
    1. Go to [supabase.com](https://supabase.com)
    2. Click "Start your project"
    3. Sign up with GitHub (free)
    4. Create a new project
    5. Choose a database password
    6. Wait 2 minutes for setup
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### Step 2: Get API Credentials
    
    1. Go to **Project Settings** → **API**
    2. Copy **Project URL** (e.g., `https://xxx.supabase.co`)
    3. Copy **anon public** API key
    4. Save both values securely
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### Step 3: Create Database Tables
    
    Go to **SQL Editor** and run this SQL:
    
    ```sql
    -- Twitter Data Table
    CREATE TABLE twitter_data (
        id TEXT PRIMARY KEY,
        text TEXT,
        created_at TIMESTAMP,
        likes INTEGER DEFAULT 0,
        retweets INTEGER DEFAULT 0,
        replies INTEGER DEFAULT 0,
        engagement INTEGER DEFAULT 0,
        sentiment NUMERIC,
        hashtags TEXT[]
    );
    
    -- Reddit Data Table
    CREATE TABLE reddit_data (
        post_id TEXT PRIMARY KEY,
        title TEXT,
        clean_title TEXT,
        author TEXT,
        subreddit TEXT,
        created_at TIMESTAMPTZ,
        score INTEGER DEFAULT 0,
        num_comments INTEGER DEFAULT 0,
        upvote_ratio NUMERIC,
        url TEXT,
        engagement INTEGER GENERATED ALWAYS AS (score + num_comments) STORED,
        synced_at TIMESTAMP DEFAULT NOW()
    );
    
    -- A/B Test Results Table
    CREATE TABLE ab_test_results (
        id SERIAL PRIMARY KEY,
        test_name TEXT,
        timestamp TIMESTAMP DEFAULT NOW(),
        variant_a TEXT,
        variant_b TEXT,
        variant_c TEXT,
        engagement_a NUMERIC,
        engagement_b NUMERIC,
        engagement_c NUMERIC,
        winner TEXT
    );
    
    -- Create indexes for faster queries
    CREATE INDEX idx_twitter_created ON twitter_data(created_at DESC);
    CREATE INDEX idx_reddit_created ON reddit_data(created_at DESC);
    CREATE INDEX idx_ab_test_timestamp ON ab_test_results(timestamp DESC);
    ```
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### Step 4: Add Credentials to Streamlit
    
    **For Local Development:**
    
    Edit `.streamlit/secrets.toml`:
    
    ```toml
    [supabase]
    url = "https://your-project.supabase.co"
    key = "your-anon-public-key"
    ```
    
    **For Streamlit Cloud Deployment:**
    
    1. Go to your app dashboard
    2. Click **Settings** → **Secrets**
    3. Paste the TOML config above
    4. Click **Save**
    5. App will restart automatically
    """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ==================== DATA SYNC ====================
st.markdown('<h3 class="section-header">🔄 Data Synchronization</h3>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown(f"""
    ### Sync Status
    
    **Last Sync:** {get_last_sync_time()}  
    **Source:** Google Sheets  
    **Destination:** Supabase PostgreSQL  
    **Frequency:** Manual (or every 10 minutes in production)
    
    **What gets synced:**
    - All Twitter posts and engagement data
    - All Reddit posts and engagement data
    - A/B test results
    - Metadata and timestamps
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    if supabase.is_connected():
        if st.button("🔄 Sync Now", key="sync_btn", use_container_width=True):
            with st.spinner("Syncing data from Google Sheets to Supabase..."):
                success, message = sync_google_sheets_to_supabase()
                
                if success:
                    st.success(message)
                    st.balloons()
                else:
                    st.error(message)
    else:
        st.warning("⚠️ Configure Supabase first to enable sync")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ==================== ARCHITECTURE DIAGRAM ====================
st.markdown('<h3 class="section-header">🏗️ Hybrid Architecture</h3>', unsafe_allow_html=True)

st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.markdown("""
```
┌─────────────────────┐
│  Data Collection    │
│    (main.py)        │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  Google Sheets      │ ← You can view/edit manually
│  (Source of Truth)  │
└──────────┬──────────┘
           │
           ↓ Sync (Manual or Auto every 10 mins)
┌─────────────────────┐
│  Supabase DB        │ ← FAST queries, no rate limits
│  (Production Cache) │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│  Streamlit App      │ ← Users get instant responses
│  (Public Website)   │
└─────────────────────┘
```

### Benefits:
- ✅ **Google Sheets:** Easy manual editing and viewing
- ✅ **Supabase:** Lightning-fast queries for users
- ✅ **Automatic Sync:** Best of both worlds
- ✅ **100% FREE:** No hosting costs
- ✅ **Unlimited Scale:** Handle 1000+ concurrent users
""")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ==================== TROUBLESHOOTING ====================
with st.expander("🔧 Troubleshooting"):
    st.markdown("""
    ### Common Issues:
    
    **1. "Supabase not connected"**
    - Check your secrets.toml file
    - Verify URL and API key are correct
    - Ensure tables are created in Supabase
    
    **2. "Sync failed"**
    - Check Google Sheets connection
    - Verify worksheet names match
    - Check Supabase table schemas
    
    **3. "No data in Supabase"**
    - Run "Sync Now" button
    - Check Google Sheets has data
    - Verify table permissions
    
    **4. "Slow queries"**
    - Data might still be in Google Sheets
    - Run sync to populate Supabase
    - Check network connection
    
    ### Need Help?
    - [Supabase Docs](https://supabase.com/docs)
    - [Streamlit Secrets](https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management)
    """)

# Footer
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 20px; color: #888;">
    <p>🗄️ Database powered by Supabase • 🔄 Automatic sync enabled • 🆓 100% FREE forever</p>
</div>
""", unsafe_allow_html=True)
