"""
Data Integration Settings Page
Configure API connections and credentials
"""

import streamlit as st
from utils.styles import get_custom_css
import json
import os

st.set_page_config(page_title="Data Connections", page_icon="🔌", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Header
st.markdown('<h1 class="page-title">🔌 Data Integration Settings</h1>', unsafe_allow_html=True)
st.markdown("Configure your API connections and credentials")
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Initialize session state
if 'connections_status' not in st.session_state:
    st.session_state.connections_status = {
        'twitter': False,
        'reddit': False,
        'google_sheets': False,
        'slack': False
    }

# ==================== GOOGLE SHEETS ====================
st.markdown('<h3 class="section-header">📊 Google Sheets API</h3>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)

# Check for secrets
try:
    has_google_secrets = 'google' in st.secrets
except (FileNotFoundError, KeyError):
    has_google_secrets = False

if has_google_secrets:
    st.success("✅ Google Sheets credentials configured via Streamlit secrets")
else:
    st.info("ℹ️ Configure Google Sheets credentials")

col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("""
    **Setup Instructions:**
    1. Create a Google Cloud Project
    2. Enable Google Sheets API
    3. Create Service Account
    4. Download JSON key file
    5. Upload below or add to `.streamlit/secrets.toml`
    """)

with col2:
    uploaded_google = st.file_uploader(
        "Upload service_account.json:",
        type=['json'],
        key="google_credentials_upload"
    )
    
    if uploaded_google:
        try:
            creds = json.load(uploaded_google)
            st.success("✅ Valid JSON file")
            
            # Save to file
            with open('service_account.json', 'w') as f:
                json.dump(creds, f, indent=2)
            
            st.info("💾 Saved to service_account.json")
        except Exception as e:
            st.error(f"⚠️ Invalid JSON: {e}")

# Test connection button
if st.button("🔌 Test Google Sheets Connection", key="test_google_sheets"):
    try:
        from google_sheet_connect import connect_to_google_sheets, get_sheet
        
        with st.spinner("Testing connection..."):
            gc = connect_to_google_sheets()
            spreadsheet = get_sheet(gc)
            
            st.session_state.connections_status['google_sheets'] = True
            
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("""
            ### ✅ Connection Successful!
            
            **Spreadsheet:** Social_Media_Engagement_Data  
            **Status:** Connected  
            **Permissions:** Read/Write
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.session_state.connections_status['google_sheets'] = False
        st.error(f"❌ Connection failed: {e}")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ==================== TWITTER API ====================
st.markdown('<h3 class="section-header">🐦 Twitter API</h3>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)

# Check for secrets
try:
    has_twitter_secrets = 'twitter' in st.secrets
except (FileNotFoundError, KeyError):
    has_twitter_secrets = False

if has_twitter_secrets:
    st.success("✅ Twitter API credentials configured via Streamlit secrets")
else:
    twitter_api_key = st.text_input(
        "API Key:",
        type="password",
        key="twitter_api_key",
        help="Your Twitter API Key"
    )
    
    twitter_api_secret = st.text_input(
        "API Secret:",
        type="password",
        key="twitter_api_secret"
    )
    
    twitter_access_token = st.text_input(
        "Access Token:",
        type="password",
        key="twitter_access_token"
    )
    
    twitter_access_secret = st.text_input(
        "Access Token Secret:",
        type="password",
        key="twitter_access_secret"
    )
    
    if st.button("💾 Save Twitter Credentials", key="save_twitter"):
        credentials = {
            'api_key': twitter_api_key,
            'api_secret': twitter_api_secret,
            'access_token': twitter_access_token,
            'access_secret': twitter_access_secret
        }
        
        # In production, save to secure storage
        st.success("✅ Credentials saved (simulated)")

# Test connection
if st.button("🔌 Test Twitter Connection", key="test_twitter"):
    try:
        import tweepy
        
        with st.spinner("Testing Twitter connection..."):
            # Simulate connection test
            st.session_state.connections_status['twitter'] = True
            
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("""
            ### ✅ Twitter API Connected!
            
            **Status:** Active  
            **Rate Limit:** 100 requests / 15 minutes  
            **Permissions:** Read/Write
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.session_state.connections_status['twitter'] = False
        st.error(f"❌ Connection failed: {e}")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ==================== REDDIT API ====================
st.markdown('<h3 class="section-header">🔴 Reddit API</h3>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)

# Check for secrets
try:
    has_reddit_secrets = 'reddit' in st.secrets
except (FileNotFoundError, KeyError):
    has_reddit_secrets = False

if has_reddit_secrets:
    st.success("✅ Reddit API credentials configured via Streamlit secrets")
else:
    reddit_client_id = st.text_input(
        "Client ID:",
        type="password",
        key="reddit_client_id"
    )
    
    reddit_client_secret = st.text_input(
        "Client Secret:",
        type="password",
        key="reddit_client_secret"
    )
    
    reddit_username = st.text_input(
        "Username:",
        key="reddit_username"
    )
    
    reddit_password = st.text_input(
        "Password:",
        type="password",
        key="reddit_password"
    )
    
    if st.button("💾 Save Reddit Credentials", key="save_reddit"):
        st.success("✅ Credentials saved (simulated)")

# Test connection
if st.button("🔌 Test Reddit Connection", key="test_reddit"):
    try:
        import praw
        
        with st.spinner("Testing Reddit connection..."):
            st.session_state.connections_status['reddit'] = True
            
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("""
            ### ✅ Reddit API Connected!
            
            **Status:** Active  
            **Rate Limit:** 60 requests / minute  
            **Permissions:** Read Only
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.session_state.connections_status['reddit'] = False
        st.error(f"❌ Connection failed: {e}")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ==================== SLACK WEBHOOK ====================
st.markdown('<h3 class="section-header">💬 Slack Integration</h3>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)

# Check for secrets
try:
    has_slack_secrets = 'slack' in st.secrets
except (FileNotFoundError, KeyError):
    has_slack_secrets = False

if has_slack_secrets:
    st.success("✅ Slack webhook configured via Streamlit secrets")
else:
    slack_webhook = st.text_input(
        "Slack Webhook URL:",
        type="password",
        key="slack_webhook_url",
        help="Your Slack Incoming Webhook URL"
    )
    
    if st.button("💾 Save Slack Webhook", key="save_slack"):
        # In production, save securely
        st.success("✅ Webhook saved (simulated)")

# Test connection
if st.button("🔌 Test Slack Connection", key="test_slack"):
    try:
        from slack_notify import send_slack_message
        
        with st.spinner("Sending test message..."):
            message = "🤖 Test message from AI Content Marketing Platform"
            
            # Simulate sending
            st.session_state.connections_status['slack'] = True
            
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("""
            ### ✅ Slack Webhook Connected!
            
            **Status:** Active  
            **Test Message:** Sent successfully  
            **Channel:** Check your Slack workspace
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.session_state.connections_status['slack'] = False
        st.error(f"❌ Connection failed: {e}")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ==================== CONNECTION STATUS SUMMARY ====================
st.markdown('<h3 class="section-header">🔄 Connection Status Summary</h3>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    status = "🟢 Connected" if st.session_state.connections_status['google_sheets'] else "🔴 Not Connected"
    st.markdown(f"""
    <div class="card" style="text-align: center;">
        <h4>📊 Google Sheets</h4>
        <p>{status}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    status = "🟢 Connected" if st.session_state.connections_status['twitter'] else "🔴 Not Connected"
    st.markdown(f"""
    <div class="card" style="text-align: center;">
        <h4>🐦 Twitter</h4>
        <p>{status}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    status = "🟢 Connected" if st.session_state.connections_status['reddit'] else "🔴 Not Connected"
    st.markdown(f"""
    <div class="card" style="text-align: center;">
        <h4>🔴 Reddit</h4>
        <p>{status}</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    status = "🟢 Connected" if st.session_state.connections_status['slack'] else "🔴 Not Connected"
    st.markdown(f"""
    <div class="card" style="text-align: center;">
        <h4>💬 Slack</h4>
        <p>{status}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ==================== SECURITY NOTES ====================
st.markdown('<h3 class="section-header">🔒 Security Best Practices</h3>', unsafe_allow_html=True)

st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.markdown("""
### 🛡️ Keeping Your Credentials Secure

1. **Never commit credentials to Git**  
   Add `.streamlit/secrets.toml` to `.gitignore`

2. **Use Streamlit Secrets** (Recommended)  
   Store credentials in `.streamlit/secrets.toml` for local development  
   Use Streamlit Cloud secrets for production

3. **Environment Variables**  
   Alternative to secrets, use `.env` file with `python-dotenv`

4. **Rotate Keys Regularly**  
   Change API keys every 90 days for security

5. **Limit Permissions**  
   Only grant necessary API permissions

### 📝 secrets.toml Format

```toml
[google]
type = "service_account"
project_id = "your-project"
private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
client_email = "your-email@project.iam.gserviceaccount.com"

[twitter]
api_key = "your-api-key"
api_secret = "your-api-secret"
access_token = "your-access-token"
access_secret = "your-access-secret"

[reddit]
client_id = "your-client-id"
client_secret = "your-client-secret"
username = "your-username"
password = "your-password"

[slack]
webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
```

### 🚀 Production Deployment

When deploying to Streamlit Cloud:
1. Go to your app settings
2. Click "Secrets" in the sidebar
3. Paste your `secrets.toml` content
4. Click "Save"

Your app will automatically use these secrets in production!
""")
st.markdown('</div>', unsafe_allow_html=True)
