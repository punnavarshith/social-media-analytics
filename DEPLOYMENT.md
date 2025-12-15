# 🚀 Deployment Guide - Streamlit Community Cloud

This guide will help you deploy your Social Media Analytics project to the cloud.

---

## 📋 Prerequisites

1. **GitHub Account** (free)
2. **Streamlit Community Cloud Account** (free) - https://share.streamlit.io/
3. **Your API Keys** ready:
   - Supabase URL & Key
   - Google Gemini API Key
   - Google Sheets Service Account JSON
   - (Optional) Twitter API keys
   - (Optional) Reddit API keys

---

## 🔧 Step 1: Push to GitHub

### 1.1 Create GitHub Repository

```bash
# Navigate to your project
cd "c:\old laptop data\INFOSYS_SPRINGBOARD_PROJECT\social_data_project"

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial deployment commit"

# Create repository on GitHub.com:
# 1. Go to https://github.com/new
# 2. Name: social-media-analytics
# 3. Public or Private (your choice)
# 4. DO NOT initialize with README

# Link and push
git remote add origin https://github.com/YOUR_USERNAME/social-media-analytics.git
git branch -M main
git push -u origin main
```

### 1.2 Verify .gitignore

Ensure these files are **NOT** pushed (they contain secrets):
- ✅ `.env` - excluded
- ✅ `service_account.json` - excluded
- ✅ `twitter_accounts.json` - excluded
- ✅ `.streamlit/secrets.toml` - excluded

---

## ☁️ Step 2: Deploy to Streamlit Cloud

### 2.1 Sign Up

1. Go to **https://share.streamlit.io/**
2. Click **"Sign up with GitHub"**
3. Authorize Streamlit to access your repositories

### 2.2 Create New App

1. Click **"New app"**
2. Select your repository: `social-media-analytics`
3. **Main file path:** `Home.py`
4. **Advanced settings:**
   - Python version: `3.11`
   - Keep defaults

### 2.3 Configure Secrets

Click **"Advanced settings" → "Secrets"** and paste this TOML configuration:

```toml
# ==================== SUPABASE ====================
[supabase]
url = "YOUR_SUPABASE_PROJECT_URL"
key = "YOUR_SUPABASE_ANON_KEY"

# ==================== GOOGLE GEMINI ====================
[gemini]
api_key = "YOUR_GEMINI_API_KEY"

# ==================== GOOGLE SHEETS ====================
# Paste your service_account.json content here
[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@your-project.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project.iam.gserviceaccount.com"

# ==================== TWITTER (OPTIONAL) ====================
[twitter]
api_key = "YOUR_TWITTER_API_KEY"
api_secret = "YOUR_TWITTER_API_SECRET"
access_token = "YOUR_TWITTER_ACCESS_TOKEN"
access_token_secret = "YOUR_TWITTER_ACCESS_TOKEN_SECRET"
bearer_token = "YOUR_TWITTER_BEARER_TOKEN"

# ==================== REDDIT (OPTIONAL) ====================
[reddit]
client_id = "YOUR_REDDIT_CLIENT_ID"
client_secret = "YOUR_REDDIT_CLIENT_SECRET"
user_agent = "social_media_analytics_bot"

# ==================== SLACK (OPTIONAL) ====================
[slack]
webhook_url = "YOUR_SLACK_WEBHOOK_URL"
```

**How to get your values:**

1. **Supabase:**
   - Go to your Supabase project → Settings → API
   - Copy URL and `anon` key

2. **Google Gemini:**
   - Go to https://makersuite.google.com/app/apikey
   - Create API key

3. **Google Sheets:**
   - Open your `service_account.json` file
   - Copy entire content into the secrets

4. **Twitter/Reddit** (optional):
   - Use your existing credentials

### 2.4 Deploy

1. Click **"Deploy"**
2. Wait 2-5 minutes for deployment
3. Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`

---

## ✅ Step 3: Verify Deployment

### 3.1 Check Data Sources

Open your deployed app and check the sidebar:

```
🗄️ Data Source Status
━━━━━━━━━━━━━━━━━━━━
Supabase: Connected ✅
🚀 Twitter: Supabase
🚀 Reddit: Supabase
```

### 3.2 Test Features

- ✅ Analytics Dashboard loads
- ✅ Charts render correctly
- ✅ Campaign Simulator works
- ✅ AI predictions function
- ✅ No error messages

---

## 🔒 Security Best Practices

### DO NOT commit these files:
- ❌ `.env`
- ❌ `service_account.json`
- ❌ `twitter_accounts.json`
- ❌ `.streamlit/secrets.toml`

### Always use Streamlit secrets for:
- ✅ API keys
- ✅ Database credentials
- ✅ OAuth tokens
- ✅ Service account files

---

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError"
**Fix:** Add missing package to `requirements.txt`

### Problem: "Supabase: Not Connected"
**Fix:** Check secrets configuration - ensure URL and key are correct

### Problem: "Google Sheets error"
**Fix:** Verify `gcp_service_account` in secrets matches your JSON file exactly

### Problem: "App crashes on startup"
**Fix:** Check Streamlit Cloud logs (click "Manage app" → "Logs")

---

## 📊 Usage Limits (Free Tier)

| Service | Free Limit | Notes |
|---------|-----------|-------|
| Streamlit Cloud | 1 public app | Unlimited usage |
| Supabase | 500MB database | 2GB bandwidth/month |
| Google Gemini | 15 req/min | 60 req/min for free tier |
| Google Sheets | 300 reads/min | Per project |

---

## 🎯 Next Steps

1. **Share your app:**
   - `https://YOUR_APP_NAME.streamlit.app`
   - Add to your resume/portfolio

2. **Monitor usage:**
   - Streamlit Cloud analytics
   - Supabase dashboard

3. **Update deployment:**
   - Push to GitHub main branch
   - Streamlit auto-redeploys

4. **Custom domain (optional):**
   - Streamlit Cloud settings → Domain

---

## 🚀 Production Deployment Checklist

- ✅ Code pushed to GitHub
- ✅ `.gitignore` protects secrets
- ✅ `requirements.txt` complete
- ✅ Secrets configured in Streamlit Cloud
- ✅ App deployed successfully
- ✅ Data sources connected
- ✅ Features tested
- ✅ Share URL with mentors/reviewers

---

**Your app is now accessible worldwide!** 🌍

Share it with:
- Your mentor for project review
- Potential employers in your portfolio
- Friends to showcase your work
