"""
Topic-Driven Content Generator
Complete replacement for old static content generation
Uses TopicPipeline for end-to-end topic-specific predictions
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import os

# Add utils to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.styles import get_custom_css
from utils.llm_topic_pipeline import LLMTopicPipeline

# Page config
st.set_page_config(
    page_title="Topic Content Generator",
    page_icon="✍️",
    layout="wide"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Header
st.markdown('<h1 class="page-title">✍️ Topic-Driven Content Generator</h1>', unsafe_allow_html=True)
st.markdown("""
**Need fresh content about a topic?** Enter any topic and get:
- 📡 Real Reddit data collection (live scraping)
- 🧠 Google Gemini trained on YOUR specific topic
- ✍️ AI-generated marketing content
- 💾 Automatic upload to Google Sheets + Supabase

**Perfect for:** Creating content strategies for products, brands, or services
""")
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Workflow explanation
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🎯 How This Works")
    st.markdown("""
    1. **Enter Topic** - Any product/brand/service
    2. **Collect Data** - Scrapes Reddit posts
    3. **Upload Data** - Saves to Sheets + Supabase
    4. **Train LLM** - Gemini learns from data
    5. **Generate** - Creates marketing content
    
    **Data Flow:**
    ```
    Reddit → Collector → Sheets → Supabase → LLM
    ```
    
    ⏱️ **Time:** 30-60 seconds for 200 posts
    """)

# ==================== STEP 1: TOPIC SELECTION ====================

st.markdown("### 🎯 Step 1: Choose Your Topic")
st.info("💡 **Tip:** This page creates NEW content from scratch by analyzing Reddit discussions. For optimizing EXISTING content, use 'Content Optimizer' page.")

col1, col2 = st.columns([3, 1])

with col1:
    # Check for existing topics
    available_topics = LLMTopicPipeline.list_available_topics()
    
    if available_topics:
        use_existing = st.checkbox("Use existing topic", value=False)
        
        if use_existing:
            selected_topic = st.selectbox(
                "Select from trained topics:",
                available_topics,
                help="Topics with existing data and trained models"
            )
        else:
            selected_topic = st.text_input(
                "Enter new topic:",
                placeholder="e.g., 'Milton bottles', 'Nike shoes', 'iPhone 15'",
                help="Enter any product, brand, or service"
            )
    else:
        selected_topic = st.text_input(
            "Enter topic:",
            placeholder="e.g., 'Milton bottles', 'Nike shoes', 'iPhone 15'",
            help="Enter any product, brand, or service"
        )
        st.info("💡 No topics trained yet. Enter a topic to start!")

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    reddit_limit = st.number_input("Reddit Posts", min_value=50, max_value=500, value=200, step=50)

# ==================== STEP 2: DATA COLLECTION & TRAINING ====================

if selected_topic:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📊 Step 2: Train Google Gemini on Topic Data")
    
    # Initialize pipeline
    if 'pipeline' not in st.session_state or st.session_state.get('current_topic') != selected_topic:
        st.session_state.pipeline = LLMTopicPipeline(selected_topic)
        st.session_state.current_topic = selected_topic
    
    pipeline = st.session_state.pipeline
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Train LLM on Topic Data", type="primary", use_container_width=True):
            with st.spinner(f"Training Google Gemini on '{selected_topic}' data..."):
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Collection
                    status_text.text("📡 Collecting topic data...")
                    progress_bar.progress(25)
                    data = pipeline.collect_data(reddit_limit=reddit_limit, youtube_limit=0, force_refresh=True)
                    
                    # Step 2: Processing
                    status_text.text("🔄 Processing & analyzing...")
                    progress_bar.progress(50)
                    processed = pipeline.process_data()
                    
                    # Step 3: Compute Statistics (This feeds the LLM)
                    status_text.text("🧠 Computing statistics for LLM...")
                    progress_bar.progress(75)
                    stats = pipeline.compute_statistics()
                    
                    progress_bar.progress(100)
                    status_text.empty()
                    
                    # Success message
                    st.success(f"✅ Google Gemini trained on '{selected_topic}' data!")
                    st.info("🤖 LLM now has context from real topic-specific posts")
                    
                    # Show results
                    result_col1, result_col2, result_col3 = st.columns(3)
                    
                    with result_col1:
                        st.metric("Posts Collected", f"{data['total_posts']}")
                    
                    with result_col2:
                        st.metric("Avg Engagement", f"{stats['engagement']['average']:.0f}")
                    
                    with result_col3:
                        sentiment = stats['sentiment']['average_polarity']
                        sentiment_label = "Positive" if sentiment > 0.1 else "Neutral" if sentiment > -0.1 else "Negative"
                        st.metric("Sentiment", sentiment_label)
                    
                    st.session_state.pipeline_ready = True
                    
                except Exception as e:
                    st.error(f"❌ Pipeline error: {e}")
                    st.info("Make sure Reddit API is configured in .env file")
    
    with col2:
        if st.button("📥 Load Existing Data", use_container_width=True):
            try:
                # Try loading cached data
                processed = pipeline.get_analytics_data()
                if processed is not None and not processed.empty:
                    st.success(f"✅ Loaded {len(processed)} posts from cache")
                    st.session_state.pipeline_ready = True
                else:
                    st.warning("⚠️ No cached data found. Run pipeline first.")
            except Exception as e:
                st.error(f"❌ Error loading data: {e}")
    
    with col3:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.session_state.pop('pipeline_ready', None)
            st.rerun()

# ==================== STEP 3: CONTENT GENERATION ====================

if selected_topic and st.session_state.get('pipeline_ready'):
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### ✍️ Step 3: Generate Content")
    
    pipeline = st.session_state.pipeline
    
    col1, col2 = st.columns(2)
    
    with col1:
        content_type = st.selectbox(
            "Content Type:",
            ["Social Media Post", "Tweet", "Blog Intro", "Product Description", "Ad Copy"]
        )
        
        prompt = st.text_area(
            "Custom Prompt (optional):",
            placeholder=f"e.g., 'Write a tweet promoting {selected_topic}'",
            height=100
        )
        
        if st.button("✨ Generate Content", type="primary"):
            with st.spinner("Generating content..."):
                if not prompt:
                    prompt = f"Write a compelling {content_type.lower()} about {selected_topic}"
                
                generated = pipeline.generate_content(prompt, max_tokens=2048)
                
                st.markdown("#### 📝 Generated Content:")
                
                # Check if content was generated
                if not generated or len(generated) < 10:
                    st.error("❌ No content generated. Please check the logs for errors.")
                    st.code(f"Received: '{generated}'")
                else:
                    # Display in a text area for better readability
                    st.text_area("Generated Content", generated, height=200, key="generated_content")
                    
                    # Also display in an expandable section for copying
                    with st.expander("📋 Copy Generated Content"):
                        st.code(generated, language="text")
                
                # Predict engagement for generated content
                prediction = pipeline.predict_engagement(generated)
                
                st.markdown("#### 📊 Predicted Engagement:")
                
                if 'error' in prediction:
                    st.error(f"⚠️ Prediction error: {prediction['error']}")
                
                pred_col1, pred_col2, pred_col3 = st.columns(3)
                
                with pred_col1:
                    engagement_value = prediction.get('predicted_engagement', 0)
                    st.metric(
                        "Predicted Engagement",
                        f"{int(engagement_value):,}",
                        help="Expected total engagement (likes + comments + shares)"
                    )
                
                with pred_col2:
                    rating = prediction.get('rating', 'Average')
                    rating_emoji = {
                        'Low': '📉',
                        'Average': '➡️',
                        'Good': '📈',
                        'High': '🔥',
                        'Viral Potential': '🚀'
                    }.get(rating, '➡️')
                    st.metric(
                        "Rating",
                        f"{rating_emoji} {rating}"
                    )
                
                with pred_col3:
                    confidence = prediction.get('confidence', 'Medium')
                    confidence_color = {
                        'Low': '🟡',
                        'Medium': '🟠',
                        'High': '🟢'
                    }.get(confidence, '🟠')
                    st.metric(
                        "Confidence",
                        f"{confidence_color} {confidence}"
                    )
                
                # Display reasoning
                if 'reasoning' in prediction and prediction['reasoning']:
                    st.markdown("**Why this score:**")
                    if isinstance(prediction['reasoning'], list):
                        for reason in prediction['reasoning']:
                            st.markdown(f"• {reason}")
                    else:
                        st.markdown(prediction['reasoning'])
                
                # Display improvement suggestion
                if 'improvement_suggestion' in prediction and prediction['improvement_suggestion']:
                    st.info(f"💡 **Improvement Tip:** {prediction['improvement_suggestion']}")
                
                # Show benchmarks
                if 'benchmark' in prediction:
                    st.markdown("**Benchmarks:**")
                    bench = prediction['benchmark']
                    st.write(f"- Average for {selected_topic}: {bench['avg_engagement']:,}")
                    st.write(f"- Top 25%: {bench['top_25_percent']:,}")
                    st.write(f"- Top 10%: {bench['top_10_percent']:,}")
    
    with col2:
        st.markdown("#### 🧪 Test Your Own Content")
        
        user_content = st.text_area(
            "Enter content to analyze:",
            placeholder=f"Write your own content about {selected_topic} here...",
            height=150
        )
        
        if st.button("🔍 Predict Engagement"):
            if user_content:
                with st.spinner("Analyzing..."):
                    prediction = pipeline.predict_engagement(user_content)
                    
                    st.markdown("#### 📈 Results:")
                    
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        st.metric(
                            "Predicted Engagement",
                            f"{prediction.get('predicted_engagement', 0):,}"
                        )
                        st.metric(
                            "Rating",
                            prediction.get('rating', 'N/A').replace('_', ' ').title()
                        )
                    
                    with result_col2:
                        st.metric(
                            "Confidence",
                            prediction.get('confidence', 'N/A').capitalize()
                        )
                        
                        # Comparison
                        avg = prediction.get('benchmark', {}).get('avg_engagement', 0)
                        pred = prediction.get('predicted_engagement', 0)
                        if avg > 0:
                            diff = ((pred - avg) / avg) * 100
                            st.metric(
                                "vs Average",
                                f"{diff:+.1f}%"
                            )
            else:
                st.warning("⚠️ Please enter some content first")

# ==================== STEP 4: ANALYTICS ====================

if selected_topic and st.session_state.get('pipeline_ready'):
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📊 Step 4: Topic Analytics")
    
    pipeline = st.session_state.pipeline
    
    try:
        df = pipeline.get_analytics_data()
        
        if df is not None and not df.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Posts", f"{len(df):,}")
            
            with col2:
                st.metric("Avg Engagement", f"{df['engagement'].mean():.0f}")
            
            with col3:
                st.metric("Top Post", f"{df['engagement'].max():.0f}")
            
            with col4:
                platforms = df['platform'].value_counts()
                st.metric("Platforms", len(platforms))
            
            # Show sample data
            with st.expander("📋 View Sample Data"):
                display_cols = ['text', 'platform', 'engagement', 'sentiment_polarity', 'created_at']
                available_cols = [col for col in display_cols if col in df.columns]
                st.dataframe(
                    df[available_cols].head(20),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("No analytics data available")
    
    except Exception as e:
        st.error(f"Error loading analytics: {e}")

# ==================== SIDEBAR INFO ====================

st.sidebar.markdown("### 💡 How It Works")

st.sidebar.markdown("""
**LLM-Based Topic Pipeline (Your Original Approach):**

1. **Enter Topic** - Any product/brand/service
2. **Collect Data** - Fetch 200+ Reddit posts about YOUR topic
3. **Compute Statistics** - Analyze engagement patterns, sentiment, timing
4. **Train LLM** - Feed stats to Google Gemini as context
5. **Generate** - LLM creates content using topic knowledge
6. **Predict** - LLM predicts engagement based on topic patterns

**Key Architecture:**
- ✅ No separate ML models
- ✅ Google Gemini learns from context
- ✅ Topic-specific data only
- ✅ Real engagement patterns
- ✅ Accurate predictions
""")

if selected_topic:
    st.sidebar.markdown(f"**Current Topic:** {selected_topic}")
    
    if st.session_state.get('pipeline_ready'):
        st.sidebar.success("✅ LLM Trained & Ready")
    else:
        st.sidebar.warning("⚠️ Train LLM first")

st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.sidebar.markdown("### 📚 Example Topics")
st.sidebar.markdown("""
- Milton water bottles
- Nike running shoes
- iPhone 15 Pro
- Starbucks coffee
- Tesla Model 3
- Sony headphones
- Amazon Echo
""")
