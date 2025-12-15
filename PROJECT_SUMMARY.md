# 📊 Social Media Data Collection & Analysis Project - Summary

**Project Name:** Social Media Data Collection, Analysis & Content Generation  
**Date Updated:** November 11, 2025  
**Status:** ✅ Production Ready - Milestones 1, 2 & 3 Complete

---

## 🎯 Project Overview

This project automatically collects data from multiple social media platforms (Twitter/X with multi-account rotation, Reddit, Google Trends), performs comprehensive exploratory data analysis (EDA) with sentiment and temporal analysis, generates optimized content using LLM with advanced parameter tuning, and provides actionable business insights. All data is stored in Google Sheets with real-time Slack notifications. The system features separate content generation and optimization workflows with both automatic and manual LLM parameter selection.

**Key Achievements:**
- ✅ **12,587 data points collected** (2,517 Twitter + 10,070 Reddit)
- ✅ **100% complete EDA** with sentiment analysis and temporal trends
- ✅ **Multi-account Twitter rotation** (14 accounts)
- ✅ **LLM-based content generation** (Ollama/Llama 3.2, free & local)
- ✅ **Content optimization** with automatic & manual LLM parameter selection
- ✅ **Dual Google Sheets storage** (Generated_Content + Optimized_Content)
- ✅ **Actionable insights discovered** (6 AM Friday = 10x better engagement)
- ✅ **A/B testing tracker** with variant metrics and winner identification
- ✅ **Real-time performance dashboard** with HTML generation
- ✅ **Automated weekly/monthly reports** with trend analysis
- ✅ **Advanced sentiment analysis** with content recommendations

---

## 📁 Project Structure

```
social_data_project/
│
├── service_account.json          ✅ Google Cloud service account credentials
├── twitter_accounts.json         ✅ Multi-account Twitter configuration (14 accounts)
├── .env                          ✅ API keys and credentials (secure)
├── .gitignore                    ✅ Protects sensitive files
├── requirements.txt              ✅ Python dependencies
│
├── google_sheet_connect.py       ✅ Google Sheets connection module
├── twitter_data.py               ✅ Twitter/X API data fetching
├── twitter_multi_account.py      ✅ Multi-account rotation system
├── reddit_data.py                ✅ Reddit API data fetching
├── trends_data.py                ✅ Google Trends data fetching
├── slack_notify.py               ✅ Slack notification module
├── write_to_sheet.py             ✅ Data writing module
├── main.py                       ✅ Main orchestration script
│
├── data_analysis.py              ✅ Comprehensive EDA with sentiment & temporal analysis
├── llm_content_generator.py      ✅ LLM content generation (saves to sheets + Slack)
├── content_optimizer.py          ✅ Content optimization (automatic & manual params + sheets + Slack)
│
├── ab_testing_tracker.py         ✅ A/B testing tracker with winner identification (Milestone 3)
├── performance_dashboard.py      ✅ Performance dashboard with HTML generation (Milestone 3)
├── automated_reports.py          ✅ Weekly/monthly automated reports (Milestone 3)
├── advanced_sentiment_analysis.py ✅ Advanced sentiment analysis system (Milestone 3)
│
├── README.md                     ✅ Setup and usage documentation
├── PROJECT_SUMMARY.md            📄 This file (technical overview)
├── MILESTONE3_GUIDE.md           ✅ Milestone 3 features guide
├── KEY_INSIGHTS.md               ✅ Data analysis findings & recommendations
├── QUICK_ACTION_CARD.md          ✅ Quick reference guide for content strategy
├── SLACK_INTEGRATION.md          ✅ Slack setup guide
├── REDDIT_SETUP.md               ✅ Reddit API setup guide
├── TWITTER_MULTI_ACCOUNT_SETUP.md ✅ Multi-account Twitter setup guide
└── OLLAMA_SETUP.md               ✅ LLM setup guide (Ollama installation)
```

---

## ✅ What's Working

### 1. **Google Sheets Integration** ✅
- Successfully connected to Google Sheets API
- Service account authentication configured
- Spreadsheet: [Social_Media_Engagement_Data](https://docs.google.com/spreadsheets/d/1VujT31YHr-gIlE2uWT6DyjPNEQXfAdmy60yTrsCAOYY)
- Automatic worksheet creation for each platform
- Data appending and formatting functionality

### 2. **Twitter/X API Integration** ✅
- **Multi-account rotation system** with 14 Twitter accounts
- Authentication successful with Twitter API v2
- Bearer token working correctly for all accounts
- Can fetch tweets by search query
- Can fetch tweets from specific users
- Automatic rate limit handling and account rotation
- Extracts: tweet text, author info, engagement metrics, timestamps
- **Achievement: 2,517 tweets collected**

### 3. **Reddit API Integration** ✅
- PRAW (Python Reddit API Wrapper) configured
- Subreddit post fetching working perfectly
- Collects: post titles, content, scores, comments, URLs
- Successfully tested with multiple subreddits
- Rate limit handling (60 requests per minute)
- **Achievement: 10,070 posts collected**

### 4. **Google Trends Integration** ✅
- Using pytrends library
- Fetches trending searches by region (US, UK, India)
- Collects search volume and trend data
- Used to discover topics for Twitter/Reddit searches
- Note: Aggressive rate limiting (~10-20 requests/hour)

### 5. **Slack Notifications** ✅
- Real-time notifications via Slack Incoming Webhooks
- Sends success notifications for each platform
- Comprehensive summary after collection completes
- Includes data counts and Google Sheet URL
- Workspace: "Social Data Collection" (#social channel)

### 6. **Data Analysis (NEW!)** ✅
- **Comprehensive EDA** (100% complete)
  - Missing value analysis
  - Outlier detection (IQR method)
  - Text cleaning and preprocessing
  - Correlation analysis
  - Statistical summaries (min, max, median, mean, std)
- **Sentiment Analysis**
  - TextBlob-based polarity scoring
  - Sentiment classification (Positive/Neutral/Negative)
  - Sentiment-engagement correlation analysis
  - **Key finding: 51.5% neutral, sentiment has minimal impact (-0.030 correlation)**
- **Temporal Trends Analysis**
  - Best posting hours detection
  - Best posting days analysis
  - Weekend vs weekday comparison
  - **Key finding: 6 AM Friday = 10x better engagement**
- **Platform Comparison**
  - **Reddit 2,053x better engagement than Twitter**
  - Optimal content length analysis (50-100 chars = +22% engagement)
  
### 7. **Content Generation** ✅
- **LLM-based content generator using Ollama**
- Powered by Llama 3.2 (free, local AI model)
- Uses real hashtags from collected data
- Loads high-performing examples from cleaned text
- Generates creative, unique posts for any topic
- **Saves to Google Sheets:** 'Generated_Content' worksheet
- **Sends Slack notifications** for each generation
- **100% free, runs locally, no API costs**
- Fulfills "Implement LLMs" requirement for Milestone 2

### 8. **Content Optimization** ✅
- **Separate optimization system** with intelligent analysis
- **Automatic LLM parameter selection** (AI-optimized)
  - Temperature: 0.4 (precise for optimization)
  - Top_p: 0.9 (consistent vocabulary)
  - Frequency penalty: 0.3 (natural flow)
  - Presence penalty: 0.2 (stay focused)
- **Manual LLM parameter selection** (user-defined)
  - Interactive prompts for each parameter
  - Validation and guidance provided
  - Custom parameter testing
- **Content analysis & scoring** (0-100 quality score)
  - Length optimization
  - Hashtag analysis
  - Engagement element detection
  - Quality rating system
- **Optimization with both parameter types**
  - Generate with automatic parameters
  - Generate with manual parameters
  - Compare both results
  - Determine winner
- **Saves to Google Sheets:** 'Optimized_Content' worksheet
  - Stores both automatic and manual results
  - Tracks parameters used
  - Records improvement scores
- **Sends Slack notifications** for optimization results
  - Shows original vs optimized content
  - Displays improvement metrics
  - Includes parameter details

### 9. **Data Pipeline** ✅
- End-to-end data collection working
- Automatic worksheet creation for each platform
- Data appending functionality
- Timestamp tracking (fetched_at column)
- Error handling throughout
- Slack notification integration
- **Total: 12,587 data points collected and analyzed**

---

## 🔧 Technical Details

### APIs Used:
- **Google Sheets API** (via gspread)
- **Twitter API v2** (via tweepy)
- **Reddit API** (via PRAW)
- **Google Trends** (via pytrends)
- **Slack Incoming Webhooks** (via requests)

### Python Version:
- Python 3.13.7

### Key Libraries:
- `gspread==6.2.1` - Google Sheets integration
- `tweepy==4.16.0` - Twitter API client
- `praw==7.8.1` - Reddit API wrapper
- `pytrends==4.9.2` - Google Trends data
- `pandas==2.2.3` - Data manipulation
- `textblob==0.18.0` - Sentiment analysis
- `python-dotenv==1.2.1` - Environment variables
- `google-auth==2.41.1` - Google authentication
- `requests==2.32.3` - Slack webhooks

### Security:
- ✅ Sensitive files in `.gitignore`
- ✅ Credentials stored in `.env` file
- ✅ Service account JSON protected
- ✅ No hardcoded API keys

---

## 📊 Data Collected

### Twitter Data Fields:
- `fetched_at` - Timestamp when data was collected
- `tweet_id` - Unique tweet identifier
- `created_at` - When the tweet was posted
- `text` - Tweet content
- `author_id` - User ID of author
- `author_username` - Twitter handle
- `author_name` - Display name
- `verified` - Verification status
- `likes` - Like count
- `retweets` - Retweet count
- `replies` - Reply count
- `language` - Tweet language
- `source` - Platform used to post

### Reddit Data Fields:
- `fetched_at` - Timestamp when data was collected
- `post_id` - Unique post identifier
- `created_at` - When the post was created
- `subreddit` - Subreddit name
- `title` - Post title
- `text` - Post content
- `author` - Reddit username
- `score` - Upvote score
- `upvote_ratio` - Ratio of upvotes to downvotes
- `num_comments` - Comment count
- `url` - Post URL
- `permalink` - Full Reddit URL

### Google Trends Data Fields:
- `fetched_at` - Timestamp when data was collected
- `query` - Search term
- `region` - Geographic region
- `value` - Search volume/interest
- `timeframe` - Time period analyzed

---

## 🚀 How to Run

### Quick Start:
```powershell
# Run data collection
python main.py

# Run data analysis (after collection)
python data_analysis.py

# Generate AI-powered content based on insights
python llm_content_generator.py
```

### Individual Component Testing:
```powershell
# Test Google Sheets connection
python google_sheet_connect.py

# Test Twitter multi-account system
python twitter_multi_account.py

# Test Reddit API
python reddit_data.py

# Test Google Trends
python trends_data.py

# Test Slack notifications
python slack_notify.py
```

---

## 🔑 Configuration

### Environment Variables (.env):
```properties
# Twitter API (14 accounts configured)
TWITTER_API_KEY=✅ Configured
TWITTER_API_SECRET=✅ Configured
TWITTER_ACCESS_TOKEN=✅ Configured
TWITTER_ACCESS_SECRET=✅ Configured
TWITTER_BEARER_TOKEN=✅ Configured

# Reddit API
REDDIT_CLIENT_ID=✅ Configured
REDDIT_CLIENT_SECRET=✅ Configured
REDDIT_USER_AGENT=✅ Configured

# Slack Webhooks
SLACK_WEBHOOK_URL=✅ Configured

# Google Sheets
GOOGLE_SHEET_ID=✅ Configured (1VujT31YHr-gIlE2uWT6DyjPNEQXfAdmy60yTrsCAOYY)
```

### Service Account:
- **Email:** `sheet-access-service@marketingaiproject-476317.iam.gserviceaccount.com`
- **Project:** marketingaiproject-476317
- **Permissions:** Editor access to Google Sheet
- **Status:** ✅ Shared and working

---

## 📈 Features

### Current Features:
- ✅ Automated data collection from Twitter (14 accounts), Reddit, Google Trends
- ✅ Multi-account Twitter rotation system to avoid rate limits
- ✅ **12,587 data points collected** (2,517 Twitter + 10,070 Reddit)
- ✅ Data storage in Google Sheets with automatic formatting
- ✅ Real-time Slack notifications during collection
- ✅ Automatic worksheet creation per platform
- ✅ Timestamp tracking for all data
- ✅ Rate limit handling for all APIs
- ✅ Comprehensive error handling and logging
- ✅ **100% complete EDA module** with sentiment & temporal analysis
- ✅ **Sentiment analysis** (polarity scoring, classification, distribution)
- ✅ **Temporal trends analysis** (best posting times/days)
- ✅ **LLM-based content generation** (Ollama/Llama 3.2, free & local)
- ✅ **A/B test variant generation** (3 tones: professional, casual, inspirational)
- ✅ **7-day content calendar** generation
- ✅ **Actionable insights** documented in KEY_INSIGHTS.md
- ✅ **Quick reference card** for content strategy (QUICK_ACTION_CARD.md)
- ✅ Modular architecture for easy expansion
- ✅ Well-documented code with comprehensive setup guides

### Key Insights Discovered:
- 📊 **Reddit 2,053x better engagement** than Twitter (5,172 vs 2.4 avg)
- ⏰ **Best posting time**: 6:00 AM on Friday (10x better engagement)
- 📏 **Optimal Reddit title length**: 50-100 characters (+22% engagement)
- � **Sentiment impact**: Minimal (-0.030 correlation) - focus on information quality
- � **Weekday advantage**: +86% better engagement than weekends
- #️⃣ **Top hashtags**: #DigitalMarketing, #SEO, #AI
- 🔑 **Top keywords**: marketing, market, media, digital, content

### Potential Enhancements:
- 🔮 Schedule automated runs (using cron/Task Scheduler)
- 🔮 Data visualization dashboard
- 🔮 Historical trend analysis over time
- 🔮 Predictive engagement modeling
- 🔮 Automated posting to platforms
- 🔮 Export insights to PDF reports

---

## 🐛 Issues Resolved

1. ✅ **Multiple Python versions conflict** - Fixed by using Python 3.13 explicitly
2. ✅ **Missing pip in Python 3.13** - Installed using `python -m ensurepip`
3. ✅ **Wrong Google Sheet ID** - Updated from private_key_id to actual sheet ID
4. ✅ **Service account not shared** - Shared sheet with service account email
5. ✅ **Environment variable caching** - Set correct variable in PowerShell session
6. ✅ **Timestamp serialization error** - Added datetime to string conversion
7. ✅ **Twitter rate limit** - Code handles automatically with wait_on_rate_limit=True

---

## ⚠️ Known Limitations

1. **Twitter API Rate Limits:**
   - Free tier: Limited requests per 15-minute window
   - Solution: Multi-account rotation system (14 accounts) successfully handles this
   - Code automatically rotates accounts when limit is reached

2. **Reddit API Rate Limits:**
   - 60 requests per minute limit
   - Read-only access with current credentials
   - Solution: Code handles limits gracefully with delays

3. **Google Trends Rate Limits:**
   - Very aggressive rate limiting (~10-20 requests/hour)
   - 429 errors common with frequent requests
   - Solution: Space out requests, use sparingly for topic discovery

4. **Google Sheets:**
   - Maximum 10 million cells per spreadsheet
   - API quotas: 300 requests per 60 seconds per user
   - Solution: Batch operations, pagination

5. **Content Generation:**
   - Currently template-based (no AI LLM)
   - Uses real hashtags from collected data
   - 100% offline, no API costs

---

## 🔐 Security Best Practices

✅ **Implemented:**
- API keys stored in `.env` file
- `.env` and `service_account.json` in `.gitignore`
- Service account with minimal required permissions
- No credentials in code or repository

⚠️ **Recommendations:**
- Rotate API keys periodically
- Monitor API usage
- Use environment-specific credentials
- Enable 2FA on developer accounts
- Review Google Cloud audit logs

---

## 📚 Documentation

- **README.md** - Complete setup guide with step-by-step instructions
- **PROJECT_SUMMARY.md** - This file (technical overview and architecture)
- **KEY_INSIGHTS.md** - Data analysis results and strategic recommendations
- **QUICK_ACTION_CARD.md** - Quick reference guide for content strategy
- **TWITTER_MULTI_ACCOUNT_SETUP.md** - Multi-account rotation setup guide
- **REDDIT_SETUP.md** - Reddit API setup guide
- **SLACK_INTEGRATION.md** - Slack webhook setup guide
- **Inline comments** - Every function documented with docstrings
- **Error messages** - Clear, actionable error messages with emoji indicators

---

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Multi-platform API integration (Twitter, Reddit, Google Trends, Slack)
- ✅ OAuth and various authentication methods
- ✅ **Multi-account rotation system** to handle rate limits
- ✅ Data collection and ETL pipeline design
- ✅ **Comprehensive exploratory data analysis (EDA)**
- ✅ **Sentiment analysis** using TextBlob
- ✅ **Temporal trends analysis** for optimal posting times
- ✅ **Statistical analysis** (correlations, outliers, distributions)
- ✅ Error handling and retry logic
- ✅ Rate limit management across different APIs
- ✅ Real-time notification systems (Slack webhooks)
- ✅ **LLM-based content generation** (Ollama/Llama 3.2)
- ✅ **Data-driven decision making** and business insights
- ✅ Modular code architecture
- ✅ Environment configuration management
- ✅ Version control best practices
- ✅ Comprehensive documentation skills

---

## 🔄 Future Roadmap

### Phase 1: ✅ COMPLETE (Milestone 1)
- Multi-platform data collection (Twitter, Reddit, Google Trends)
- Multi-account Twitter rotation system (14 accounts)
- Google Sheets integration
- Slack notification system
- Comprehensive error handling
- **Achievement: 12,587 data points collected**

### Phase 2: ✅ COMPLETE (Milestone 2)
- Comprehensive EDA with 100% coverage
- Sentiment analysis (TextBlob integration)
- Temporal trends analysis (best posting times/days)
- Statistical analysis (correlations, outliers)
- **LLM-based content generation (Ollama/Llama 3.2)**
  - Separate generation file with Google Sheets integration
  - Slack notifications for generated content
- **Content optimization system**
  - Automatic LLM parameter selection (AI-optimized)
  - Manual LLM parameter selection (user-defined)
  - Content analysis & quality scoring (0-100)
  - Comparison between automatic vs manual optimization
  - Separate Google Sheets storage for optimized content
  - Slack notifications for optimization results
- **Dual worksheet system:**
  - 'Generated_Content' - All generated posts
  - 'Optimized_Content' - All optimizations with parameters
- Actionable insights documentation
- **Achievement: Free local AI, Separate generation/optimization, Parameter comparison, Reddit 2,053x better, 6 AM Friday optimal**

### Phase 3: 🔮 Planned (Milestone 3)
- Scheduled automated runs (daily/weekly)
- Data visualization dashboard
- Performance tracking over time
- Predictive engagement modeling
- Automated posting to platforms
- Real-time monitoring dashboard

### Phase 4: 🔮 Future
- Advanced ML models for engagement prediction
- Multi-platform cross-posting automation
- Real-time trend detection
- Competitive analysis features
- Custom reporting and export options

---

## 📞 Support & Resources

### Official Documentation:
- [Google Sheets API](https://developers.google.com/sheets/api)
- [Twitter API v2](https://developer.twitter.com/en/docs/twitter-api)
- [Reddit API](https://www.reddit.com/dev/api/)
- [Google Trends (pytrends)](https://pypi.org/project/pytrends/)
- [Slack Incoming Webhooks](https://api.slack.com/messaging/webhooks)
- [Tweepy Documentation](https://docs.tweepy.org/)
- [PRAW Documentation](https://praw.readthedocs.io/)
- [gspread Documentation](https://docs.gspread.org/)

### Project Links:
- **Google Sheet:** [Social_Media_Engagement_Data](https://docs.google.com/spreadsheets/d/1VujT31YHr-gIlE2uWT6DyjPNEQXfAdmy60yTrsCAOYY)
- **Slack Workspace:** Social Data Collection (#social channel)
- **Twitter Developer Portal:** [developer.twitter.com](https://developer.twitter.com/)
- **Reddit Apps:** [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
- **Google Cloud Console:** [console.cloud.google.com](https://console.cloud.google.com/)

### Key Documentation:
- **KEY_INSIGHTS.md** - Critical findings and recommendations
- **QUICK_ACTION_CARD.md** - Quick reference for content strategy

---

## ✨ Conclusion

This project successfully demonstrates a production-ready multi-platform social media data collection, analysis, and content generation pipeline. The system collected **12,587 data points**, performed comprehensive EDA (100% coverage), discovered actionable insights (Reddit 2,053x better, 6 AM Friday optimal), and generates AI-powered content using free local LLMs. The multi-account Twitter rotation system, advanced analytics capabilities, and Ollama-based content generation make this a complete end-to-end solution.

**Status:** ✅ **PRODUCTION READY - MILESTONES 1 & 2 COMPLETE**

**Platforms:** Twitter (14 accounts) ✅ | Reddit ✅ | Google Trends ✅ | Slack ✅

**Analytics:** EDA ✅ | Sentiment Analysis ✅ | Temporal Trends ✅ | LLM Content Generation ✅

**Key Achievement:** Discovered that Reddit provides 2,053x better engagement, posting at 6 AM Friday yields 10x better results, and implemented free local AI (Ollama/Llama 3.2) for creative content generation.

---

**Built with ❤️ using Python, Google Sheets API, Twitter API, Reddit API, Google Trends, TextBlob, Slack, and Ollama (Llama 3.2)**

*Last Updated: November 9, 2025*
