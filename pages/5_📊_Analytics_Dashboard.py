"""
Analytics Dashboard Page
Visualize performance metrics and insights
"""

import streamlit as st
from utils.styles import get_custom_css
from utils.backend import load_twitter_data, load_reddit_data, get_last_30_days_data
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from textblob import TextBlob

st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Header
st.markdown('<h1 class="page-title">📊 Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown("Comprehensive performance analytics and insights")
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Filter options
st.sidebar.markdown("### 🎯 Filters")
date_range = st.sidebar.selectbox(
    "Date Range:",
    ["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"],
    index=1
)

platform_filter = st.sidebar.multiselect(
    "Platforms:",
    ["Twitter", "Reddit"],
    default=["Twitter", "Reddit"]
)

# Determine days parameter for data loading
days_to_load = None  # Default: all time
if date_range == "Last 7 Days":
    days_to_load = 7
elif date_range == "Last 30 Days":
    days_to_load = 30
elif date_range == "Last 90 Days":
    days_to_load = 90
# else: days_to_load remains None (all time)

# Load data with dynamic date range
twitter_df = load_twitter_data(days=days_to_load)
reddit_df = load_reddit_data(days=days_to_load)

# Convert created_at to datetime for both dataframes
for df in [twitter_df, reddit_df]:
    if 'created_at' in df.columns and len(df) > 0:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True)

# Debug validator function
def debug_validate(df, name):
    st.sidebar.markdown(f"### 🔍 {name} Validation")
    if len(df) > 0 and 'created_at' in df.columns:
        st.sidebar.write(f"Rows: {len(df)}")
        st.sidebar.write(f"Min date: {df['created_at'].min()}")
        st.sidebar.write(f"Max date: {df['created_at'].max()}")
        st.sidebar.write(f"Source: {df.attrs.get('source', 'unknown')}")
    else:
        st.sidebar.write(f"No data for {name}")
    st.sidebar.markdown("---")

# Data Source Status Panel
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🗄️ Data Source Status")
    
    try:
        from utils.supabase_db import get_supabase_client
        supabase = get_supabase_client()
        if supabase.is_connected():
            st.success("Supabase: Connected ✅")
        else:
            st.error("Supabase: Not Connected ❌")
    except:
        st.error("Supabase: Not Available ❌")
    
    # Show data source for each platform
    if hasattr(twitter_df, 'attrs') and "source" in twitter_df.attrs:
        source_icon = "🚀" if twitter_df.attrs["source"] == "Supabase" else "📄"
        st.caption(f"{source_icon} Twitter: {twitter_df.attrs['source']}")
    else:
        st.caption("❓ Twitter: Unknown source")
    
    if hasattr(reddit_df, 'attrs') and "source" in reddit_df.attrs:
        source_icon = "🚀" if reddit_df.attrs["source"] == "Supabase" else "📄"
        st.caption(f"{source_icon} Reddit: {reddit_df.attrs['source']}")
    else:
        st.caption("❓ Reddit: Unknown source")
    
    st.markdown("---")
    st.markdown("### � Debug Counts")
    st.write(f"Twitter rows: {len(twitter_df)}")
    st.write(f"Reddit rows: {len(reddit_df)}")
    st.write(f"Twitter source: {twitter_df.attrs.get('source', 'unknown')}")
    st.write(f"Reddit source: {reddit_df.attrs.get('source', 'unknown')}")
    
    st.markdown("---")

# Call debug validators
debug_validate(twitter_df, "Twitter")
debug_validate(reddit_df, "Reddit")

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Data Info")
    st.caption(f"Date range: {date_range}")
    st.caption(f"Days param: {days_to_load if days_to_load else 'All Time'}")
    st.caption(f"Twitter rows loaded: {len(twitter_df):,}")
    st.caption(f"Reddit rows loaded: {len(reddit_df):,}")
    st.warning("⚠️ If row counts look wrong, clear cache below!")
    if st.button("🔄 Clear Cache & Reload", type="primary"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("✅ Cache cleared! Reloading...")
        st.rerun()

# Convert created_at to datetime for both dataframes
for df in [twitter_df, reddit_df]:
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True)

# Debug validator function
def debug_validate(df, name):
    st.sidebar.markdown(f"### 🔍 {name} Validation")
    if len(df) > 0 and 'created_at' in df.columns:
        st.sidebar.write(f"Rows: {len(df)}")
        st.sidebar.write(f"Min date: {df['created_at'].min()}")
        st.sidebar.write(f"Max date: {df['created_at'].max()}")
        st.sidebar.write(f"Source: {df.attrs.get('source', 'unknown')}")
    else:
        st.sidebar.write(f"No data for {name}")
    st.sidebar.markdown("---")

# Data is already pre-filtered by days_to_load, so use directly
twitter_filtered = twitter_df
reddit_filtered = reddit_df

# Ensure engagement columns exist in filtered data
if len(twitter_filtered) > 0:
    if 'engagement' not in twitter_filtered.columns:
        twitter_filtered['likes'] = pd.to_numeric(twitter_filtered.get('likes', 0), errors='coerce').fillna(0)
        twitter_filtered['retweets'] = pd.to_numeric(twitter_filtered.get('retweets', 0), errors='coerce').fillna(0)
        twitter_filtered['replies'] = pd.to_numeric(twitter_filtered.get('replies', 0), errors='coerce').fillna(0)
        twitter_filtered['engagement'] = twitter_filtered['likes'] + twitter_filtered['retweets'] + twitter_filtered['replies']
    twitter_filtered['engagement'] = pd.to_numeric(twitter_filtered['engagement'], errors='coerce').fillna(0)

if len(reddit_filtered) > 0:
    if 'engagement' not in reddit_filtered.columns:
        reddit_filtered['score'] = pd.to_numeric(reddit_filtered.get('score', 0), errors='coerce').fillna(0)
        reddit_filtered['num_comments'] = pd.to_numeric(reddit_filtered.get('num_comments', 0), errors='coerce').fillna(0)
        reddit_filtered['engagement'] = reddit_filtered['score'] + reddit_filtered['num_comments']
    reddit_filtered['engagement'] = pd.to_numeric(reddit_filtered['engagement'], errors='coerce').fillna(0)

# ==================== KEY METRICS ====================
st.markdown('<h3 class="section-header">📈 Key Performance Metrics</h3>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_posts = 0
    if "Twitter" in platform_filter:
        total_posts += len(twitter_filtered)
    if "Reddit" in platform_filter:
        total_posts += len(reddit_filtered)
    
    st.metric("📝 Total Posts", f"{total_posts:,}")

with col2:
    total_engagement = 0
    if "Twitter" in platform_filter and len(twitter_filtered) > 0:
        total_engagement += twitter_filtered['engagement'].sum()
    if "Reddit" in platform_filter and len(reddit_filtered) > 0:
        total_engagement += reddit_filtered['engagement'].sum()
    
    st.metric("💬 Total Engagement", f"{int(total_engagement):,}")

with col3:
    avg_engagement = 0
    if total_posts > 0:
        avg_engagement = total_engagement / total_posts
    
    st.metric("📊 Avg Engagement", f"{avg_engagement:.1f}")

with col4:
    max_engagement = 0
    if "Twitter" in platform_filter and len(twitter_filtered) > 0:
        max_engagement = max(max_engagement, twitter_filtered['engagement'].max())
    if "Reddit" in platform_filter and len(reddit_filtered) > 0:
        max_engagement = max(max_engagement, reddit_filtered['engagement'].max())
    
    st.metric("🏆 Max Engagement", f"{int(max_engagement)}")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ==================== TWITTER ANALYTICS ====================
if "Twitter" in platform_filter and len(twitter_filtered) > 0:
    st.markdown('<h3 class="section-header">🐦 Twitter Analytics</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Ensure engagement is numeric
        if 'engagement' in twitter_filtered.columns:
            twitter_filtered['engagement'] = pd.to_numeric(twitter_filtered['engagement'], errors='coerce').fillna(0)
        
        # Top posts
        st.markdown("**🏆 Top Performing Posts**")
        top_posts = twitter_filtered.nlargest(10, 'engagement')[['text', 'likes', 'retweets', 'engagement']].copy()
        top_posts['text'] = top_posts['text'].str[:60] + '...'
        st.dataframe(top_posts, use_container_width=True, hide_index=True)
    
    with col2:
        # Ensure numeric columns
        twitter_filtered['likes'] = pd.to_numeric(twitter_filtered['likes'], errors='coerce').fillna(0)
        twitter_filtered['retweets'] = pd.to_numeric(twitter_filtered['retweets'], errors='coerce').fillna(0)
        twitter_filtered['replies'] = pd.to_numeric(twitter_filtered['replies'], errors='coerce').fillna(0)
        
        # Engagement breakdown
        total_likes = twitter_filtered['likes'].sum()
        total_retweets = twitter_filtered['retweets'].sum()
        total_replies = twitter_filtered['replies'].sum()
        
        fig = go.Figure(data=[go.Pie(
            labels=['Likes', 'Retweets', 'Replies'],
            values=[total_likes, total_retweets, total_replies],
            marker_colors=['#e74c3c', '#2ecc71', '#3498db'],
            hole=0.4
        )])
        fig.update_layout(
            title='Engagement Type Distribution',
            template='plotly_dark',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Timeline
    st.markdown("**📈 Engagement Over Time**")
    
    daily_engagement = twitter_filtered.groupby(twitter_filtered['created_at'].dt.date)['engagement'].agg(['sum', 'mean', 'count']).reset_index()
    daily_engagement.columns = ['Date', 'Total', 'Average', 'Count']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_engagement['Date'],
        y=daily_engagement['Total'],
        mode='lines+markers',
        name='Total Engagement',
        line=dict(color='#667eea', width=3),
        fill='tozeroy'
    ))
    fig.update_layout(
        title='Daily Total Engagement',
        template='plotly_dark',
        height=350,
        xaxis_title='Date',
        yaxis_title='Engagement'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Best posting times
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**⏰ Best Posting Hours**")
        
        twitter_filtered['hour'] = twitter_filtered['created_at'].dt.hour
        hourly = twitter_filtered.groupby('hour')['engagement'].mean().reset_index()
        hourly = hourly.sort_values('engagement', ascending=False)
        
        fig = px.bar(
            hourly,
            x='hour',
            y='engagement',
            title='Average Engagement by Hour',
            color='engagement',
            color_continuous_scale='Viridis',
            labels={'hour': 'Hour of Day', 'engagement': 'Avg Engagement'}
        )
        fig.update_layout(template='plotly_dark', height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**📅 Best Posting Days**")
        
        twitter_filtered['day_name'] = twitter_filtered['created_at'].dt.day_name()
        daily = twitter_filtered.groupby('day_name')['engagement'].mean().reset_index()
        
        # Order days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily['day_name'] = pd.Categorical(daily['day_name'], categories=day_order, ordered=True)
        daily = daily.sort_values('day_name')
        
        fig = px.bar(
            daily,
            x='day_name',
            y='engagement',
            title='Average Engagement by Day',
            color='engagement',
            color_continuous_scale='Plasma',
            labels={'day_name': 'Day', 'engagement': 'Avg Engagement'}
        )
        fig.update_layout(template='plotly_dark', height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    # Hashtag analysis
    st.markdown("**#️⃣ Top Hashtags**")
    
    if 'text' in twitter_filtered.columns:
        hashtags = []
        for text in twitter_filtered['text']:
            hashtags.extend([word for word in str(text).split() if word.startswith('#')])
        
        if hashtags:
            from collections import Counter
            hashtag_counts = Counter(hashtags).most_common(10)
            
            df_hashtags = pd.DataFrame(hashtag_counts, columns=['Hashtag', 'Count'])
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(df_hashtags, use_container_width=True, hide_index=True)
            
            with col2:
                fig = px.bar(
                    df_hashtags,
                    x='Hashtag',
                    y='Count',
                    title='Most Used Hashtags',
                    color='Count',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(template='plotly_dark', height=300)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hashtags found in posts")
    
    # Emoji analysis
    st.markdown("**😊 Emoji vs Non-Emoji Performance**")
    
    if 'text' in twitter_filtered.columns:
        def has_emoji(text):
            emoji_pattern = '[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]'
            import re
            return bool(re.search(emoji_pattern, str(text)))
        
        twitter_filtered['has_emoji'] = twitter_filtered['text'].apply(has_emoji)
        
        emoji_comparison = twitter_filtered.groupby('has_emoji')['engagement'].mean().reset_index()
        emoji_comparison['Category'] = emoji_comparison['has_emoji'].map({True: 'With Emoji', False: 'Without Emoji'})
        
        fig = px.bar(
            emoji_comparison,
            x='Category',
            y='engagement',
            title='Emoji Impact on Engagement',
            color='engagement',
            color_continuous_scale='Viridis',
            text='engagement'
        )
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig.update_layout(template='plotly_dark', height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        with_emoji_avg = emoji_comparison[emoji_comparison['has_emoji'] == True]['engagement'].values
        without_emoji_avg = emoji_comparison[emoji_comparison['has_emoji'] == False]['engagement'].values
        
        if len(with_emoji_avg) > 0 and len(without_emoji_avg) > 0:
            improvement = ((with_emoji_avg[0] / without_emoji_avg[0]) - 1) * 100
            
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown(f"""
            **💡 Emoji Impact:**
            - Posts with emojis: {with_emoji_avg[0]:.1f} avg engagement
            - Posts without emojis: {without_emoji_avg[0]:.1f} avg engagement
            - **{improvement:+.1f}% difference**
            """)
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ==================== REDDIT ANALYTICS ====================
if "Reddit" in platform_filter and len(reddit_filtered) > 0:
    st.markdown('<h3 class="section-header">🔴 Reddit Analytics</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Posts", f"{len(reddit_filtered):,}")
    
    with col2:
        reddit_filtered['score'] = pd.to_numeric(reddit_filtered['score'], errors='coerce').fillna(0)
        st.metric("Avg Score", f"{reddit_filtered['score'].mean():.1f}")
    
    with col3:
        reddit_filtered['num_comments'] = pd.to_numeric(reddit_filtered['num_comments'], errors='coerce').fillna(0)
        st.metric("Avg Comments", f"{reddit_filtered['num_comments'].mean():.1f}")
    
    # Top posts
    st.markdown("**🏆 Top Reddit Posts**")
    if 'title' in reddit_filtered.columns and len(reddit_filtered) > 0:
        top_reddit = reddit_filtered.nlargest(min(10, len(reddit_filtered)), 'engagement')[['title', 'score', 'num_comments', 'engagement']].copy()
        top_reddit['title'] = top_reddit['title'].str[:80] + '...'
        st.dataframe(top_reddit, use_container_width=True, hide_index=True)
    else:
        st.info("No Reddit posts available for the selected date range")

# ==================== EXPORT OPTIONS ====================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<h3 class="section-header">📥 Export & Reports</h3>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Export CSV", use_container_width=True):
        st.info("✅ CSV export would be generated here")

with col2:
    if st.button("📄 Generate PDF Report", use_container_width=True):
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("""
        ### ✅ PDF Report Generated (Simulated)
        
        **Report Contents:**
        - Executive Summary
        - Key Metrics Dashboard
        - Top Performing Content
        - Best Posting Times
        - Hashtag Analysis
        - Recommendations
        
        *In production, this would generate a downloadable PDF file.*
        """)
        st.markdown('</div>', unsafe_allow_html=True)

with col3:
    if st.button("📧 Email Report", use_container_width=True):
        st.success("✅ Report would be emailed")

# ==================== INSIGHTS ====================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<h3 class="section-header">💡 AI-Generated Insights</h3>', unsafe_allow_html=True)

if len(twitter_filtered) > 0:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    
    best_hour = twitter_filtered.groupby('hour')['engagement'].mean().idxmax()
    best_day = twitter_filtered.groupby('day_name')['engagement'].mean().idxmax()
    
    st.markdown(f"""
    ### 🎯 Key Insights
    
    1. **Optimal Posting Time:** Your best performing hour is **{best_hour}:00**
    2. **Best Day:** **{best_day}** shows highest average engagement
    3. **Content Strategy:** Posts with emojis perform {improvement:.0f}% better
    4. **Posting Frequency:** You're posting **{len(twitter_filtered) / 30:.1f} times per day** on average
    
    ### 📈 Recommendations
    
    - Focus posting during peak hours ({best_hour}:00 ± 2 hours)
    - Increase content volume on {best_day}
    - Continue using emojis for visual appeal
    - Test different content formats during off-peak hours
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("📊 No data available. Run data collection to see insights!")
