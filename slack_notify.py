"""
Slack Notification Module
Sends notifications to Slack channel about data collection status
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def send_slack_message(message, emoji=":robot_face:"):
    """
    Send a message to Slack channel
    
    Args:
        message (str): The message to send
        emoji (str): Emoji icon for the message
    
    Returns:
        bool: True if successful, False otherwise
    """
    webhook_url = os.getenv('SLACK_WEBHOOK_URL')
    
    if not webhook_url or webhook_url == 'your_slack_webhook_url_here':
        print("⚠️ Slack webhook URL not configured")
        return False
    
    try:
        payload = {
            "text": message,
            "icon_emoji": emoji
        }
        
        response = requests.post(webhook_url, json=payload)
        
        if response.status_code == 200:
            # Show first line of message without truncation
            first_line = message.split('\n')[0].strip()
            print(f"✅ Slack notification sent: {first_line}")
            return True
        else:
            print(f"❌ Failed to send Slack notification: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error sending Slack notification: {e}")
        return False


def send_data_collection_summary(twitter_count=0, reddit_count=0, instagram_count=0, 
                                 facebook_count=0, trends_count=0, sheet_url=None):
    """
    Send a formatted summary of data collection to Slack
    
    Args:
        twitter_count: Number of tweets collected
        reddit_count: Number of Reddit posts collected
        instagram_count: Number of Instagram posts collected
        facebook_count: Number of Facebook posts collected
        trends_count: Number of Google Trends items collected
        sheet_url: URL to the Google Sheet
    
    Returns:
        bool: True if successful
    """
    total = twitter_count + reddit_count + instagram_count + facebook_count + trends_count
    
    message = f"""
📊 *Social Media Data Collection Complete!*

*Data Collected:*
• Twitter: {twitter_count} tweets
• Reddit: {reddit_count} posts
• Instagram: {instagram_count} posts
• Facebook: {facebook_count} posts
• Google Trends: {trends_count} items

*Total: {total} items collected*
"""
    
    if sheet_url:
        message += f"\n📈 View data: {sheet_url}"
    
    return send_slack_message(message, emoji=":chart_with_upwards_trend:")


def send_error_alert(platform, error_message):
    """
    Send an error alert to Slack
    
    Args:
        platform (str): The platform where error occurred
        error_message (str): The error message
    
    Returns:
        bool: True if successful
    """
    message = f"""
⚠️ *Error in Data Collection*

*Platform:* {platform}
*Error:* {error_message}

Please check the logs for more details.
"""
    
    return send_slack_message(message, emoji=":warning:")


def send_success_notification(platform, count):
    """
    Send a success notification for a specific platform
    
    Args:
        platform (str): The platform name
        count (int): Number of items collected
    
    Returns:
        bool: True if successful
    """
    platform_emojis = {
        'Twitter': ':bird:',
        'Reddit': ':reddit:',
        'Instagram': ':camera:',
        'Facebook': ':facebook:',
        'Google Trends': ':chart_with_upwards_trend:'
    }
    
    emoji = platform_emojis.get(platform, ':white_check_mark:')
    message = f"✅ {platform}: Successfully collected {count} items"
    
    return send_slack_message(message, emoji=emoji)


if __name__ == "__main__":
    # Test the Slack integration
    print("Testing Slack integration...")
    print("=" * 60)
    
    # Test 1: Simple message
    print("\n1. Sending test message...")
    send_slack_message("🚀 Hello from Social Data Collector! Slack integration is working!")
    
    # Test 2: Data collection summary
    print("\n2. Sending sample data collection summary...")
    send_data_collection_summary(
        twitter_count=10,
        reddit_count=15,
        instagram_count=0,
        facebook_count=0,
        trends_count=0,
        sheet_url="https://docs.google.com/spreadsheets/d/1VujT31YHr-gIlE2uWT6DyjPNEQXfAdmy60yTrsCAOYY"
    )
    
    print("\n" + "=" * 60)
    print("✅ Check your Slack #social channel for messages!")
