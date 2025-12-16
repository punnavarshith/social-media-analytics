"""
Campaign Simulator Page
Simulate multi-post campaigns and optimize schedules
"""

import streamlit as st
from utils.styles import get_custom_css
from utils.backend import get_simulator, load_twitter_data, load_reddit_data
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Campaign Simulator", page_icon="📋", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Header
st.markdown('<h1 class="page-title">📋 Campaign Simulator</h1>', unsafe_allow_html=True)
st.markdown("Plan and optimize your multi-post social media campaigns")
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Initialize
simulator = get_simulator()

# ==================== INPUT METHODS ====================
st.markdown('<h3 class="section-header">📝 Campaign Setup</h3>', unsafe_allow_html=True)

input_method = st.radio(
    "Input Method:",
    ["Manual Entry", "CSV Upload"],
    horizontal=True,
    key="campaign_input_method"
)

platform = st.selectbox(
    "🎯 Target Platform:",
    ["twitter", "reddit"],
    key="campaign_platform"
)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

posts_to_simulate = []

# ==================== MANUAL ENTRY ====================
if input_method == "Manual Entry":
    st.markdown('<h3 class="section-header">✍️ Enter Posts</h3>', unsafe_allow_html=True)
    
    num_posts = st.slider(
        "Number of posts:",
        min_value=2,
        max_value=10,
        value=3,
        key="num_posts_slider"
    )
    
    for i in range(num_posts):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            post_content = st.text_area(
                f"Post {i+1}:",
                height=80,
                key=f"campaign_post_{i}",
                placeholder=f"Enter content for post {i+1}..."
            )
            if post_content:
                posts_to_simulate.append(post_content)
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"**Post {i+1}**")
            st.markdown(f"📏 {len(post_content)} chars")

# ==================== CSV UPLOAD ====================
else:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    **📄 CSV Format:**
    - Single column named `content`
    - Each row = one post
    
    **Example:**
    ```
    content
    Check out our new feature! 🚀
    Learn how to boost engagement today
    Join our community of 10k+ users
    ```
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload campaign posts CSV:",
        type=['csv'],
        key="campaign_csv"
    )
    
    if uploaded_file:
        try:
            df_upload = pd.read_csv(uploaded_file)
            
            if 'content' in df_upload.columns:
                posts_to_simulate = df_upload['content'].tolist()
                st.success(f"✅ Loaded {len(posts_to_simulate)} posts")
                st.dataframe(df_upload, use_container_width=True)
            else:
                st.error("⚠️ CSV must have a 'content' column")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ==================== SIMULATION OPTIONS ====================
st.markdown('<h3 class="section-header">🎛️ Simulation Options</h3>', unsafe_allow_html=True)

simulation_type = st.radio(
    "Select simulation type:",
    ["Optimal Timing", "Best Platform", "Full 7-Day Schedule"],
    horizontal=True,
    key="simulation_type"
)

col1, col2 = st.columns(2)

with col1:
    campaign_days = st.slider(
        "Campaign Duration (days):",
        min_value=1,
        max_value=30,
        value=7,
        key="campaign_days"
    )

with col2:
    start_date = st.date_input(
        "Start Date:",
        value=datetime.now(),
        key="campaign_start_date"
    )

# ==================== RUN SIMULATION ====================
run_simulation_btn = st.button(
    "🚀 Run Simulation",
    type="primary",
    use_container_width=True,
    key="run_campaign_simulation"
)

# Initialize session state for simulation results
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None

if run_simulation_btn:
    if len(posts_to_simulate) < 2:
        st.error("⚠️ Please enter at least 2 posts to simulate a campaign!")
    else:
        with st.spinner(f"🔮 Simulating {simulation_type.lower()}..."):
            
            # ==================== OPTIMAL TIMING ====================
            if simulation_type == "Optimal Timing":
                results = []
                
                for i, content in enumerate(posts_to_simulate, 1):
                    timing_result = simulator.simulate_timing_variants(
                        content,
                        platform=platform,
                        days=min(campaign_days, 7)
                    )
                    
                    if timing_result and timing_result.get('optimal_time'):
                        optimal = timing_result['optimal_time']
                        
                        # Use correct key names from campaign_simulator.py
                        if optimal and 'optimal_day' in optimal and 'optimal_hour' in optimal:
                            results.append({
                                'Post': f'Post {i}',
                                'Content': content[:50] + '...' if len(content) > 50 else content,
                                'Optimal Day': optimal['optimal_day'],
                                'Optimal Hour': f"{optimal['optimal_hour']}:00",
                                'Predicted Engagement': optimal['predicted_engagement']
                            })
                        else:
                            # Fallback if keys are missing
                            st.warning(f"⚠️ Post {i}: Incomplete timing data, skipping...")
                            continue
                
                # Store results in session state
                if results:
                    st.session_state.simulation_results = {
                        'type': 'optimal_timing',
                        'data': results,
                        'platform': platform
                    }
                    st.success(f"✅ Simulation complete! {len(results)} posts analyzed.")
            
            # ==================== BEST PLATFORM ====================
            elif simulation_type == "Best Platform":
                results = []
                
                for i, content in enumerate(posts_to_simulate, 1):
                    platform_result = simulator.simulate_platform_comparison(content)
                    
                    if platform_result is not None and len(platform_result) > 0:
                        best_platform = platform_result.iloc[0]
                        results.append({
                            'Post': f'Post {i}',
                            'Content': content[:50] + '...' if len(content) > 50 else content,
                            'Best Platform': best_platform['platform'],
                            'Predicted Engagement': best_platform['predicted_engagement']
                        })
                
                # Store results in session state
                if results:
                    st.session_state.simulation_results = {
                        'type': 'best_platform',
                        'data': results
                    }
                    st.success(f"✅ Simulation complete! {len(results)} posts analyzed.")
            
            # ==================== FULL 7-DAY SCHEDULE ====================
            else:
                optimal_times = simulator.optimize_posting_schedule(
                    num_posts=len(posts_to_simulate),
                    duration_days=campaign_days
                )
                
                if optimal_times:
                    schedule = []
                    
                    for i, (post, time) in enumerate(zip(posts_to_simulate, optimal_times), 1):
                        schedule.append({
                            'Post #': i,
                            'Content': post[:60] + '...' if len(post) > 60 else post,
                            'Date': time.strftime('%Y-%m-%d'),
                            'Day': time.strftime('%A'),
                            'Time': time.strftime('%H:%M'),
                            'Full DateTime': time
                        })
                    
                    # Store results in session state
                    st.session_state.simulation_results = {
                        'type': 'full_schedule',
                        'data': schedule,
                        'campaign_days': campaign_days
                    }
                    st.success(f"✅ Simulation complete! {len(schedule)} posts scheduled.")

# ==================== DISPLAY SIMULATION RESULTS ====================
if st.session_state.simulation_results is not None:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">📊 Simulation Results</h3>', unsafe_allow_html=True)
    
    result_type = st.session_state.simulation_results['type']
    result_data = st.session_state.simulation_results['data']
    
    # ==================== DISPLAY OPTIMAL TIMING ====================
    if result_type == 'optimal_timing':
        df_results = pd.DataFrame(result_data)
        
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown(f"""
        ### ✅ Optimal Schedule Generated
        
        **📊 Total Posts:** {len(result_data)}  
        **📈 Average Predicted Engagement:** {df_results['Predicted Engagement'].mean():.1f}  
        **🏆 Best Performing Post:** {df_results.loc[df_results['Predicted Engagement'].idxmax(), 'Post']}
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.dataframe(df_results, use_container_width=True, hide_index=True)
        
        # Chart
        fig = px.bar(
            df_results,
            x='Post',
            y='Predicted Engagement',
            color='Predicted Engagement',
            color_continuous_scale='Viridis',
            title='Predicted Engagement by Post',
            text='Predicted Engagement'
        )
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # ==================== DISPLAY BEST PLATFORM ====================
    elif result_type == 'best_platform':
        df_results = pd.DataFrame(result_data)
        
        platform_counts = df_results['Best Platform'].value_counts()
        recommended_platform = platform_counts.index[0]
        
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown(f"""
        ### 🎯 Recommended Platform: {recommended_platform.upper()}
        
        **📊 Analysis:**
        - {platform_counts.get('twitter', 0)} posts perform better on Twitter
        - {platform_counts.get('reddit', 0)} posts perform better on Reddit
        
        **📈 Average Engagement:** {df_results['Predicted Engagement'].mean():.1f}
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.dataframe(df_results, use_container_width=True, hide_index=True)
        
        # Platform distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                values=platform_counts.values,
                names=platform_counts.index,
                title='Best Platform Distribution',
                color_discrete_sequence=['#667eea', '#764ba2']
            )
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                df_results,
                x='Post',
                y='Predicted Engagement',
                color='Best Platform',
                title='Engagement by Post & Platform',
                text='Predicted Engagement'
            )
            fig.update_traces(texttemplate='%{text:.1f}')
            fig.update_layout(template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
    
    # ==================== DISPLAY FULL SCHEDULE ====================
    elif result_type == 'full_schedule':
        df_schedule = pd.DataFrame(result_data)
        campaign_days = st.session_state.simulation_results.get('campaign_days', 7)
        
        # Calculate actual span
        actual_span = (df_schedule['Full DateTime'].max() - df_schedule['Full DateTime'].min()).days
        
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown(f"""
        ### 📅 {campaign_days}-Day Campaign Schedule
        
        **📊 Total Posts:** {len(df_schedule)}  
        **📆 Start Date:** {df_schedule['Date'].min()}  
        **📆 End Date:** {df_schedule['Date'].max()}  
        **📏 Actual Span:** {actual_span} days
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.dataframe(
            df_schedule[['Post #', 'Content', 'Date', 'Day', 'Time']],
            use_container_width=True,
            hide_index=True
        )
        
        # Timeline visualization
        fig = px.scatter(
            df_schedule,
            x='Full DateTime',
            y='Post #',
            color='Day',
            title='Campaign Timeline',
            labels={'Full DateTime': 'Schedule', 'Post #': 'Post Number'},
            size=[10]*len(df_schedule)
        )
        fig.update_layout(template='plotly_dark', height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Day distribution
        day_dist = df_schedule['Day'].value_counts().reset_index()
        day_dist.columns = ['Day', 'Count']
        
        fig = px.bar(
            day_dist,
            x='Day',
            y='Count',
            title='Posts by Day of Week',
            color='Count',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(template='plotly_dark', height=350)
        st.plotly_chart(fig, use_container_width=True)

# ==================== HISTORICAL DATA ====================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<h3 class="section-header">📊 Historical Performance Data</h3>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Twitter Analytics", "Reddit Analytics"])

with tab1:
    twitter_df = load_twitter_data()
    
    # Convert created_at to datetime
    if 'created_at' in twitter_df.columns:
        twitter_df['created_at'] = pd.to_datetime(twitter_df['created_at'], errors='coerce', utc=True)
    
    if len(twitter_df) > 0:
        # Ensure all numeric columns exist and are properly typed
        if 'likes' not in twitter_df.columns:
            twitter_df['likes'] = 0
        if 'retweets' not in twitter_df.columns:
            twitter_df['retweets'] = 0
        if 'replies' not in twitter_df.columns:
            twitter_df['replies'] = 0
            
        twitter_df['likes'] = pd.to_numeric(twitter_df['likes'], errors='coerce').fillna(0)
        twitter_df['retweets'] = pd.to_numeric(twitter_df['retweets'], errors='coerce').fillna(0)
        twitter_df['replies'] = pd.to_numeric(twitter_df['replies'], errors='coerce').fillna(0)
        
        # Create or fix engagement column
        if 'engagement' not in twitter_df.columns:
            twitter_df['engagement'] = twitter_df['likes'] + twitter_df['retweets'] + twitter_df['replies']
        else:
            twitter_df['engagement'] = pd.to_numeric(twitter_df['engagement'], errors='coerce').fillna(0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Posts", f"{len(twitter_df):,}")
        with col2:
            st.metric("Avg Engagement", f"{twitter_df['engagement'].mean():.1f}")
        with col3:
            st.metric("Max Engagement", f"{int(twitter_df['engagement'].max())}")
        
        # Top posts
        st.markdown("**🏆 Top Performing Posts:**")
        if 'text' in twitter_df.columns and len(twitter_df) > 0:
            top_posts = twitter_df.nlargest(min(5, len(twitter_df)), 'engagement')[['text', 'engagement']].copy()
            top_posts['text'] = top_posts['text'].str[:80] + '...'
            st.dataframe(top_posts, use_container_width=True, hide_index=True)
        else:
            st.info("Text column not available in data")
    else:
        st.info("No Twitter data available")

with tab2:
    reddit_df = load_reddit_data()
    
    # Convert created_at to datetime
    if 'created_at' in reddit_df.columns:
        reddit_df['created_at'] = pd.to_datetime(reddit_df['created_at'], errors='coerce', utc=True)
    
    if len(reddit_df) > 0:
        # Ensure all numeric columns exist and are properly typed
        if 'score' not in reddit_df.columns:
            reddit_df['score'] = 0
        if 'num_comments' not in reddit_df.columns:
            reddit_df['num_comments'] = 0
            
        reddit_df['score'] = pd.to_numeric(reddit_df['score'], errors='coerce').fillna(0)
        reddit_df['num_comments'] = pd.to_numeric(reddit_df['num_comments'], errors='coerce').fillna(0)
        
        # Create or fix engagement column
        if 'engagement' not in reddit_df.columns:
            reddit_df['engagement'] = reddit_df['score'] + reddit_df['num_comments']
        else:
            reddit_df['engagement'] = pd.to_numeric(reddit_df['engagement'], errors='coerce').fillna(0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Posts", f"{len(reddit_df):,}")
        with col2:
            st.metric("Avg Engagement", f"{reddit_df['engagement'].mean():.1f}")
        with col3:
            st.metric("Max Engagement", f"{int(reddit_df['engagement'].max())}")
    else:
        st.info("No Reddit data available")
