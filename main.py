"""
Main Script - Social Media Data Collection
Fetches data from multiple platforms and writes to Google Sheets:
- Twitter
- Reddit
- Google Trends
Uses Google Trends to discover trending topics, then searches Twitter/Reddit for those topics
"""

import os
import sys
import time
import pandas as pd

# Fix encoding for Windows console (only when not running in Streamlit)
if sys.platform == 'win32' and not hasattr(sys.stdout, '_is_wrapped'):
    import codecs
    # Check if stdout has buffer attribute (terminal mode)
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

try:
    from dotenv import load_dotenv
except Exception:
    # python-dotenv not available in the environment; provide a no-op fallback
    def load_dotenv():
        return None

# Import custom modules
from google_sheet_connect import connect_to_google_sheets, get_sheet
from twitter_data import fetch_tweets
from twitter_multi_account import authenticate_twitter_multi
from reddit_data import authenticate_reddit, search_reddit
from trends_data import authenticate_trends, get_trending_searches
from write_to_sheet import (
    write_twitter_data, 
    write_reddit_data,
    write_trends_data,
    format_worksheet
)
from slack_notify import send_data_collection_summary, send_success_notification

# Load environment variables
load_dotenv()

# Configuration - Set to False to skip Twitter due to rate limits
ENABLE_TWITTER = os.getenv('ENABLE_TWITTER', 'false').lower() == 'true'
ENABLE_REDDIT = os.getenv('ENABLE_REDDIT', 'true').lower() == 'true'


def main():
    """
    Main function to orchestrate the data collection and writing process
    Enhanced to use Google Trends topics as input for Twitter/Reddit searches
    """
    print("=" * 60)
    print("🚀 Starting Social Media Data Collection (Enhanced)")
    print("=" * 60)
    
    # Track counts for Slack notification
    twitter_count = 0
    reddit_count = 0
    trends_count = 0
    
    # Step 1: Connect to Google Sheets
    print("\n📊 Step 1: Connecting to Google Sheets...")
    client = connect_to_google_sheets()
    if not client:
        print("❌ Failed to connect to Google Sheets. Exiting...")
        return
    
    spreadsheet = get_sheet(client)
    if not spreadsheet:
        print("❌ Failed to open spreadsheet. Exiting...")
        return
    
    # Step 2: Fetch Google Trends Data (FIRST - to discover trending topics)
    print("\n� Step 2: Fetching Google Trends data (to discover trending topics)...")
    
    trending_topics = []
    
    try:
        # Authenticate with Google Trends
        pytrends = authenticate_trends()
        
        if not pytrends:
            raise Exception("Failed to authenticate with Google Trends")
        
        # Get trending searches from multiple regions for MORE topics
        print("   🌎 Getting trends from United States...")
        trends_us = get_trending_searches(pytrends, country='united_states')
        
        time.sleep(3)  # Avoid rate limit
        
        print("   🌍 Getting trends from United Kingdom...")
        trends_uk = get_trending_searches(pytrends, country='united_kingdom')
        
        time.sleep(3)
        
        print("   🌏 Getting trends from India...")
        trends_india = get_trending_searches(pytrends, country='india')
        
        # Combine all trending topics
        all_trends = []
        if trends_us is not None and not trends_us.empty:
            all_trends.extend(trends_us['trending_query'].tolist()[:10])
            trends_count += len(trends_us)
        if trends_uk is not None and not trends_uk.empty:
            all_trends.extend(trends_uk['trending_query'].tolist()[:10])
        if trends_india is not None and not trends_india.empty:
            all_trends.extend(trends_india['trending_query'].tolist()[:10])
        
        # Add marketing-related base topics to ensure relevant data
        base_topics = [
            "AI marketing",
            "social media trends",
            "content marketing",
            "digital marketing",
            "advertising",
            "SEO",
            "email marketing",
            "influencer marketing",
            "brand strategy",
            "marketing automation"
        ]
        
        # Combine and deduplicate
        trending_topics = list(set(all_trends + base_topics))
        
        print(f"\n✅ Found {len(trending_topics)} unique topics to search!")
        print(f"📋 Sample topics: {', '.join(trending_topics[:8])}...")
        
        # Write Trends data to Google Sheets
        if trends_us is not None and not trends_us.empty:
            print("\n📝 Writing Google Trends data to Google Sheets...")
            success = write_trends_data(spreadsheet, trends_us)
            if success:
                send_success_notification('Google Trends', trends_count)
                try:
                    worksheet = spreadsheet.worksheet("Google Trends Data")
                    format_worksheet(worksheet)
                except:
                    pass
        
    except Exception as e:
        print(f"⚠️ Error fetching Google Trends: {e}")
        print("⚠️ Using default marketing topics instead...")
        trending_topics = [
            "marketing", "advertising", "social media", "content creation",
            "digital marketing", "SEO", "email marketing", "branding",
            "AI marketing", "influencer marketing"
        ]
    
    # Step 3: Fetch Twitter Data (using trending topics)
    print(f"\n🐦 Step 3: Fetching Twitter data for topics...")
    
    if not ENABLE_TWITTER:
        print("⚠️ Twitter collection is DISABLED (set ENABLE_TWITTER=true in .env to enable)")
        print("⚠️ Reason: Twitter Free API has strict rate limits for topic searches")
        twitter_count = 0
    else:
        twitter_manager = authenticate_twitter_multi(skip_accounts=['Account_1'])
    
        all_twitter_data = []
    
        if twitter_manager and trending_topics:
            # Limit topics to number of available accounts (14)
            num_accounts = len(twitter_manager.clients)
            topics_to_search = trending_topics[:num_accounts]
            
            print(f"📊 Using {num_accounts} accounts for {len(topics_to_search)} topics (1 topic per account)\n")
        
            for i, topic in enumerate(topics_to_search):
                account_name = twitter_manager.get_current_account_name()
                print(f"   [{i+1}/{len(topics_to_search)}] {account_name} → Searching for: '{topic}'")
            
                try:
                    # Get current client (don't rotate on failure, just skip)
                    client = twitter_manager.get_current_client()
                    
                    # Fetch tweets for this topic
                    twitter_df = fetch_tweets(
                        client,
                        query=f"{topic} -is:retweet",
                        max_results=100
                    )
                    
                    if twitter_df is not None and not twitter_df.empty:
                        all_twitter_data.append(twitter_df)
                        print(f"      ✓ Found {len(twitter_df)} tweets")
                    else:
                        print(f"      ⚠️ No tweets found for '{topic}'")
                        
                except Exception as e:
                    print(f"      ⚠️ Error with '{topic}': {e}")
                
                # Move to next account for next topic
                if i < len(topics_to_search) - 1:  # Don't rotate after last topic
                    twitter_manager.rotate_account()
            
            # Combine all Twitter data
            if all_twitter_data:
                combined_twitter = pd.concat(all_twitter_data, ignore_index=True)
                twitter_count = len(combined_twitter)
                
                print(f"\n✅ Total Twitter data collected: {twitter_count} tweets")
                
                # Write Twitter data to Google Sheets
                print("\n📝 Writing Twitter data to Google Sheets...")
                success = write_twitter_data(spreadsheet, combined_twitter)
            
                if success:
                    send_success_notification('Twitter', twitter_count)
                    try:
                        worksheet = spreadsheet.worksheet("Twitter Data")
                        format_worksheet(worksheet)
                    except:
                        pass
            else:
                print("⚠️ No Twitter data collected")
        else:
            print("⚠️ Skipping Twitter data collection")
    
    # Step 4: Fetch Reddit Data (using trending topics)
    print(f"\n🔴 Step 4: Fetching Reddit data for topics...")
    
    if not ENABLE_REDDIT:
        print("⚠️ Reddit collection is DISABLED (set ENABLE_REDDIT=true in .env to enable)")
        reddit_count = 0
    else:
        reddit_client = authenticate_reddit()
        all_reddit_data = []
    
        if reddit_client and trending_topics:
            topics_to_search = trending_topics[:14]  # Match Twitter: 14 topics
        
            for i, topic in enumerate(topics_to_search, 1):
                print(f"   [{i}/{len(topics_to_search)}] Searching Reddit for: '{topic}'")
            
                try:
                    reddit_df = search_reddit(
                        reddit_client,
                        query=topic,
                        limit=100  # Increased to 100 posts per topic (was 50)
                    )
                    
                    if reddit_df is not None and not reddit_df.empty:
                        all_reddit_data.append(reddit_df)
                        print(f"      ✓ Found {len(reddit_df)} posts")
                    
                    time.sleep(2)  # Avoid rate limit
                    
                except Exception as e:
                    print(f"      ⚠️ Error searching '{topic}': {e}")
            
            # Combine all Reddit data
            if all_reddit_data:
                combined_reddit = pd.concat(all_reddit_data, ignore_index=True)
                reddit_count = len(combined_reddit)
                
                print(f"\n✅ Total Reddit data collected: {reddit_count} posts")
                
                # Write Reddit data to Google Sheets
                print("\n📝 Writing Reddit data to Google Sheets...")
                success = write_reddit_data(spreadsheet, combined_reddit)
                
                if success:
                    send_success_notification('Reddit', reddit_count)
                    try:
                        worksheet = spreadsheet.worksheet("Reddit Data")
                        format_worksheet(worksheet)
                    except:
                        pass
            else:
                print("⚠️ No Reddit data collected")
        else:
            print("⚠️ Skipping Reddit data collection")
    
    # Final Summary
    print("\n📈 Step 5: Generating summary...")
    
    # Summary complete
    print("\n" + "=" * 60)
    print("✅ Social Media Data Collection Complete!")
    print("=" * 60)
    print(f"📊 Spreadsheet: {spreadsheet.title}")
    print(f"🔗 URL: {spreadsheet.url}")
    print("\n📋 Data Collected:")
    print(f"   🐦 Twitter: {twitter_count} tweets")
    print(f"   🔴 Reddit: {reddit_count} posts")
    print(f"   📈 Trending Topics: {trends_count} topics analyzed")
    print(f"\n🎯 Total Data Points: {twitter_count + reddit_count}")
    print(f"🔍 Topics Searched: {len(trending_topics)} unique topics")
    print("\nCheck your Google Sheet for all collected data!")
    
    # Send Slack notification summary
    print("\n📬 Sending Slack notification...")
    send_data_collection_summary(
        twitter_count=twitter_count,
        reddit_count=reddit_count,
        instagram_count=0,
        facebook_count=0,
        trends_count=trends_count,
        sheet_url=spreadsheet.url
    )


if __name__ == "__main__":
    main()
