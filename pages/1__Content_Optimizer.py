"""
Content Optimizer & Variant Generator
Optimize and improve your existing social media content with AI-powered variants
"""

import streamlit as st
from utils.styles import get_custom_css
from utils.advanced_content_optimizer import AdvancedContentOptimizer
from textblob import TextBlob
import pandas as pd

st.set_page_config(page_title="Content Optimizer", page_icon="📝", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Header
st.markdown('<h1 class="page-title">📝 Content Optimizer & Variant Generator</h1>', unsafe_allow_html=True)
st.markdown("""
**Already have content?** Paste it here and get:
- 🎨 Multiple optimized variants (emojis, hashtags, CTAs)
- 📊 Engagement predictions for each variant
- 🤖 AI-powered improvements

**Perfect for:** Refining tweets, posts, or captions before publishing
""")
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Workflow explanation
with st.expander("📋 How It Works", expanded=False):
    st.markdown("""
    ### Workflow:
    1. **Paste Your Content** ✍️ - Enter the text you already wrote
    2. **Generate Variants** 🔄 - Get multiple optimized versions
    3. **Compare Predictions** 📊 - See engagement scores for each
    4. **Pick the Best** ✅ - Choose the highest-performing variant
    
    **Note:** This page does NOT collect new data. It uses cached marketing data for predictions.
    """)

# Initialize optimizer
optimizer = AdvancedContentOptimizer()

# Show status in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🔧 System Status")
    
    # Check Gemini status
    gemini_status = optimizer.check_gemini_status()
    
    if gemini_status['available']:
        st.success(f"✅ Google Gemini - Running")
        st.caption(f"Model: {optimizer.model}")
        if not gemini_status['target_model_available']:
            st.warning(f"⚠️ Model '{optimizer.model}' not found")
            st.caption("Check your model name")
    else:
        st.error("❌ Google Gemini - Not Configured")
        st.caption("Check secrets.toml")
        st.warning("⚠️ **Variants require Gemini API**")
    
    st.markdown("### 💡 Tips")
    st.markdown("""
    - **Variants = Gemini API only**
    - Paid Tier 1: 100 req/min
    - Add topic for better results
    - Hook variant = stronger opening
    - Complete = full rewrite
    - Scores: 0-100 normalized
    """)

# Initialize session state
if 'generated_variants' not in st.session_state:
    st.session_state.generated_variants = []
if 'original_content' not in st.session_state:
    st.session_state.original_content = ""
if 'topic_context' not in st.session_state:
    st.session_state.topic_context = ""

# ==================== INPUT SECTION ====================
st.markdown('<h3 class="section-header">✍️ Step 1: Paste Your Content</h3>', unsafe_allow_html=True)
st.info("⚠️ **Important:** This is for EXISTING content you already wrote. For creating NEW content from scratch, use 'Topic Content Generator' page.")

col1, col2 = st.columns([3, 1])

with col1:
    original_content = st.text_area(
        "Paste your existing content below:",
        height=150,
        placeholder="Example: 'Excited to announce our new product launch! 🚀'\n\nPaste YOUR content here (tweet, post, caption, etc.)",
        key="content_input",
        value=st.session_state.original_content
    )

with col2:
    # Topic context (optional but recommended)
    topic_context = st.text_input(
        "📌 Topic/Product (optional):",
        placeholder="e.g., 'OnePlus15', 'Tesla'",
        help="Provide topic for smarter optimization",
        key="topic_input",
        value=st.session_state.topic_context
    )
    
    platform = st.selectbox(
        "🎯 Target Platform:",
        ["twitter", "reddit"],
        key="platform_select"
    )
    
    num_variants = st.slider(
        "Number of variants:",
        min_value=2,
        max_value=5,
        value=3,
        key="num_variants"
    )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ==================== ACTION BUTTONS ====================
st.markdown('<h3 class="section-header">🎬 Step 2: Generate & Optimize</h3>', unsafe_allow_html=True)

# Check Gemini status for button display
gemini_check = optimizer.check_gemini_status()
if not gemini_check['available']:
    st.warning("⚠️ **Google Gemini is not configured!** Add google_api_key to secrets.toml")

col1, col2 = st.columns(2)

with col1:
    generate_btn = st.button("🚀 Generate Smart Variants", type="primary", use_container_width=True, key="generate_variants_btn")

with col2:
    analyze_only_btn = st.button("📊 Analyze Content Only", use_container_width=True, key="analyze_btn")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ==================== ANALYZE CONTENT ====================
if analyze_only_btn and original_content:
    st.markdown('<h3 class="section-header">📊 Content Analysis</h3>', unsafe_allow_html=True)
    
    analysis = optimizer.analyze_content(original_content)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sentiment_label = "Positive ✅" if analysis['sentiment_polarity'] > 0.1 else "Negative ❌" if analysis['sentiment_polarity'] < -0.1 else "Neutral ⚪"
        st.metric("Sentiment", sentiment_label, f"{analysis['sentiment_polarity']:.2f}")
    
    with col2:
        readability_label = "Easy ✅" if analysis['readability'] > 60 else "Hard ❌" if analysis['readability'] < 40 else "Medium ⚪"
        st.metric("Readability", readability_label, f"{analysis['readability']:.0f}")
    
    with col3:
        st.metric("Word Count", analysis['word_count'])
    
    with col4:
        st.metric("Engagement Features", f"{analysis['emoji_count']}📱 {analysis['hashtag_count']}#️⃣")
    
    # Feature checklist
    st.markdown("**Feature Checklist:**")
    features = {
        "Has CTA": analysis['has_cta'],
        "Has Question": analysis['has_question'],
        "Has Emojis": analysis['emoji_count'] > 0,
        "Has Hashtags": analysis['hashtag_count'] > 0,
    }
    
    for feature, has_it in features.items():
        st.markdown(f"{'✅' if has_it else '❌'} {feature}")

# ==================== GENERATE VARIANTS ====================
if generate_btn and original_content:
    st.session_state.original_content = original_content
    st.session_state.topic_context = topic_context
    
    # Check Gemini status first
    with st.spinner("🔍 Checking Google Gemini status..."):
        gemini_status = optimizer.check_gemini_status()
    
    if not gemini_status['available']:
        st.error(f"❌ {gemini_status['message']}")
        st.warning("**⚠️ Content variants can ONLY be generated by Google Gemini API**")
        
        st.info("""
        **To configure Gemini:**
        1. Get API key from https://aistudio.google.com/apikey
        2. Add to `.streamlit/secrets.toml`:
           ```
           google_api_key = "YOUR_API_KEY"
           ```
        3. Restart Streamlit
        4. Come back and click "Generate Smart Variants" again
        """)
        
        if gemini_status['status'] == 'not_configured':
            st.code('google_api_key = "YOUR_API_KEY"', language="toml")
    
    elif not gemini_status['target_model_available']:
        st.error(f"❌ Model '{optimizer.model}' is not available")
        st.info(f"**Available models:** {', '.join(gemini_status['models'][:5])}")
        st.warning(f"**Update model name in optimizer initialization**")
    
    else:
        # Gemini is ready - generate variants
        st.success(f"✅ {gemini_status['message']}")
        # Show actual rate limit from optimizer
        rate_delay = optimizer._rate_limit_delay
        if rate_delay <= 0.5:
            st.info(f"⏱️ Rate limiting: {rate_delay}s between requests (Paid Tier 1 - 100 req/min)")
        else:
            st.info(f"⏱️ Rate limiting: {rate_delay}s between requests (Free Tier - 15 req/min)")
        
        with st.spinner(f"🤖 Generating AI-powered variants with {optimizer.model}..."):
            # Generate smart variants
            result = optimizer.generate_smart_variants(
                content=original_content,
                platform=platform.lower(),
                topic=topic_context if topic_context else None,
                num_variants=num_variants
            )
        
        if not result['success']:
            st.error(f"❌ Failed to generate variants: {result['error']}")
            if result.get('failed_variants'):
                st.warning(f"Failed variants: {', '.join(result['failed_variants'])}")
        else:
            variants = result['variants']
            
            if result.get('failed_variants'):
                st.warning(f"⚠️ Some variants failed to generate: {', '.join(result['failed_variants'])}")
                st.info(f"Successfully generated {len(variants)} out of {num_variants} variants")
            else:
                st.success(f"✅ Successfully generated {len(variants)} variants!")
            
            # Analyze original content
            original_analysis = optimizer.analyze_content(original_content)
            original_score = optimizer.calculate_engagement_score(original_content, original_analysis, platform)
            
            # Build results table
            variant_results = []
            
            # Add original
            variant_results.append({
                'Variant': 'Original',
                'Content': original_content,
                'Modification': 'No changes',
                'Predicted Engagement': int(original_score),
                'Sentiment': f"{original_analysis['sentiment_polarity']:.2f}",
                'Readability': f"{original_analysis['readability']:.0f}",
                'Word Count': original_analysis['word_count']
            })
            
            # Add variants
            for i, variant in enumerate(variants, 1):
                variant_analysis = optimizer.analyze_content(variant['content'])
                variant_score = optimizer.calculate_engagement_score(variant['content'], variant_analysis, platform)
                
                variant_results.append({
                    'Variant': f"Variant {i}: {variant['name']}",
                    'Content': variant['content'],
                    'Modification': variant['modification'],
                    'Predicted Engagement': int(variant_score),
                    'Sentiment': f"{variant_analysis['sentiment_polarity']:.2f}",
                    'Readability': f"{variant_analysis['readability']:.0f}",
                    'Word Count': variant_analysis['word_count']
                })
        
            st.session_state.generated_variants = variant_results

# ==================== DISPLAY RESULTS ====================
if st.session_state.generated_variants:
    st.markdown('<h3 class="section-header">📊 Step 3: Compare Variants</h3>', unsafe_allow_html=True)
    
    df = pd.DataFrame(st.session_state.generated_variants)
    
    # Sort by engagement score
    df = df.sort_values('Predicted Engagement', ascending=False)
    
    # Winner determination with 3% threshold
    winner = df.iloc[0]
    runner_up = df.iloc[1] if len(df) > 1 else None
    
    # Calculate if there's a decisive winner (3% threshold)
    has_decisive_winner = False
    gap_percentage = 0
    
    if runner_up is not None:
        engagement_gap = winner['Predicted Engagement'] - runner_up['Predicted Engagement']
        gap_percentage = (engagement_gap / max(runner_up['Predicted Engagement'], 1)) * 100
        has_decisive_winner = gap_percentage >= 3.0
    
    # Display winner or similarity notice
    if has_decisive_winner:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown(f"""
        ### 🏆 Best Performing Variant: {winner['Variant']}
        
        **📊 Engagement Score (Relative):** {winner['Predicted Engagement']:.0f}/100  
        **😊 Sentiment:** {winner['Sentiment']} (polarity)  
        **📖 Readability:** {winner['Readability']} (Flesch score)  
        **📝 Words:** {winner['Word Count']}
        
        **✨ What Changed:** {winner['Modification']}
        
        **📈 Performance Gap:** {gap_percentage:.1f}% ahead of {runner_up['Variant']} (Decisive winner)
        
        **Content:**
        ```
        {winner['Content']}
        ```
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box" style="background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%); padding: 20px; border-radius: 10px; color: white;">', unsafe_allow_html=True)
        st.markdown(f"""
        ### ⚖️ Statistically Similar Performance
        
        **Top Variant:** {winner['Variant']}  
        **📊 Engagement Score:** {winner['Predicted Engagement']:.0f}/100  
        **📉 Performance Gap:** {gap_percentage:.1f}% (below 3% threshold)
        
        **✨ What Changed:** {winner['Modification']}
        
        **💡 Insight:** Variants perform similarly - no decisive winner.  
        All variants are equally viable. Consider other factors (brand voice, clarity, length) when choosing.
        
        **Content:**
        ```
        {winner['Content']}
        ```
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Display all results with side-by-side comparison
    st.markdown("### 📋 All Variants with Change Highlights")
    
    for idx, row in df.iterrows():
        with st.expander(f"{row['Variant']} - Score: {row['Engagement Score']:.0f}", expanded=(idx == df.index[0])):
            # Modification description
            st.info(f"**What changed:** {row['Modification']}")
            
            # Side-by-side comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Content:**")
                st.text_area(
                    "original_view",
                    st.session_state.get('original_content', ''),
                    height=150,
                    key=f"orig_{idx}",
                    label_visibility="collapsed"
                )
            
            with col2:
                st.markdown(f"**{row['Variant']}:**")
                st.text_area(
                    "variant_view",
                    row['Content'],
                    height=150,
                    key=f"var_{idx}",
                    label_visibility="collapsed"
                )
            
            # Metrics
            st.markdown("**Key Metrics:**")
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                st.metric("Engagement Score", f"{row['Engagement Score']:.0f}/100")
            with metrics_col2:
                sentiment_label = "Positive" if float(row['Sentiment']) > 0.1 else "Neutral" if float(row['Sentiment']) > -0.1 else "Negative"
                st.metric("Sentiment", sentiment_label)
            with metrics_col3:
                st.metric("Readability", f"{row['Readability']}")
            with metrics_col4:
                st.metric("Word Count", row['Word Count'])
            
            # Length change indicator (using word count, not char count)
            orig_words = len(st.session_state.get('original_content', '').split())
            var_words = row['Word Count']
            length_change = ((var_words - orig_words) / orig_words * 100) if orig_words > 0 else 0
            
            # Updated thresholds: 70-130% is acceptable
            if length_change < -30 or length_change > 30:
                st.warning(f"⚠️ Length changed by {length_change:+.1f}% (original: {orig_words} words, variant: {var_words} words)")
            else:
                st.success(f"✅ Length change: {length_change:+.1f}% (within ±30% target)")
            
            # Copy button
            st.markdown("**Ready to copy:**")
            st.code(row['Content'], language=None)
            if st.button(f"📋 Copy {row['Variant']}", key=f"copy_{idx}"):
                st.success(f"✅ {row['Variant']} ready to copy from the code block above!")
    
    # Chart comparison
    st.markdown("---")
    st.markdown("### 📊 Engagement Score Comparison")
    
    import plotly.express as px
    
    fig = px.bar(
        df,
        x='Variant',
        y='Engagement Score',
        color='Engagement Score',
        color_continuous_scale='Viridis',
        title='Relative Engagement Score Comparison',
        text='Engagement Score',
        labels={'Engagement Score': 'Engagement Score (0-100)'}
    )
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(showlegend=False, height=400)
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed comparison table
    st.markdown("### 📈 Detailed Metrics Table")
    st.dataframe(
        df[['Variant', 'Engagement Score', 'Sentiment', 'Readability', 'Word Count', 'Modification']],
        use_container_width=True,
        hide_index=True
    )

elif original_content:
    st.info("👆 Click 'Generate Smart Variants' to create AI-optimized content variations with LLM rewriting!")
else:
    st.warning("👈 Paste your content in Step 1 to get started")
