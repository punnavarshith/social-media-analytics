"""
A/B Testing Lab Page
Compare content variants and track results
"""

import streamlit as st
from utils.styles import get_custom_css
from utils.backend import get_predictor, write_to_google_sheets, send_slack_notification
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="A/B Testing Lab", page_icon="🧪", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Header
st.markdown('<h1 class="page-title">🧪 A/B Testing Lab</h1>', unsafe_allow_html=True)
st.markdown("Compare content variants using AI-powered relative engagement scoring")

# Transparency notice
st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.markdown("""
**🧠 About Our Scoring System:**
This system uses **Relative Engagement Scores (0-100)** for comparative analysis, not real-world engagement predictions.
Scores are generated through context-aware inference using topic-level statistical signals from your 1,761 Reddit and 2,143 Twitter posts.

✅ Use for: Comparing variants to find the strongest approach  
❌ Not for: Predicting exact likes, comments, or upvotes
""")
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Initialize predictor
predictor = get_predictor()

# Initialize session state
if 'ab_test_results' not in st.session_state:
    st.session_state.ab_test_results = None

# ==================== INPUT SECTION ====================
st.markdown('<h3 class="section-header">📝 Test Setup</h3>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    input_method = st.radio(
        "Input Method:",
        ["Manual Entry", "File Upload"],
        horizontal=True,
        key="input_method"
    )

with col2:
    platform = st.selectbox(
        "🎯 Platform:",
        ["twitter", "reddit"],
        key="ab_platform"
    )
    
    test_name = st.text_input(
        "Test Name:",
        placeholder="e.g., Product Launch Campaign",
        key="test_name"
    )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ==================== MANUAL ENTRY ====================
if input_method == "Manual Entry":
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🅰️ Variant A")
        variant_a = st.text_area(
            "Enter Variant A:",
            height=150,
            placeholder="Type your first content variant here...",
            key="variant_a",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🅱️ Variant B")
        variant_b = st.text_area(
            "Enter Variant B:",
            height=150,
            placeholder="Type your second content variant here...",
            key="variant_b",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Optional: Additional variants
    st.markdown("---")
    add_more = st.checkbox("➕ Add More Variants (C, D, E)", key="add_more_variants")
    
    variant_c = ""
    variant_d = ""
    variant_e = ""
    
    if add_more:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            variant_c = st.text_area("🅲 Variant C:", height=100, key="variant_c")
        with col2:
            variant_d = st.text_area("🅳 Variant D:", height=100, key="variant_d")
        with col3:
            variant_e = st.text_area("🅴 Variant E:", height=100, key="variant_e")

# ==================== FILE UPLOAD ====================
else:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    **📄 CSV Format Requirements:**
    - Column 1: `variant_name` (e.g., Variant A, Variant B)
    - Column 2: `content` (the post text)
    
    **Example:**
    ```
    variant_name,content
    Variant A,Check out our new product! 🚀
    Variant B,Discover our amazing new product today!
    ```
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload CSV file:",
        type=['csv'],
        key="ab_test_csv"
    )
    
    if uploaded_file:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df_upload)} variants from file")
            st.dataframe(df_upload, use_container_width=True)
        except Exception as e:
            st.error(f"⚠️ Error reading file: {e}")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ==================== RUN TEST BUTTON ====================
run_test_btn = st.button(
    "🚀 Run A/B Test",
    type="primary",
    use_container_width=True,
    key="run_ab_test"
)

# ==================== RUN TEST ====================
if run_test_btn:
    variants_to_test = []
    
    if input_method == "Manual Entry":
        if variant_a:
            variants_to_test.append({"name": "Variant A", "content": variant_a})
        if variant_b:
            variants_to_test.append({"name": "Variant B", "content": variant_b})
        if variant_c:
            variants_to_test.append({"name": "Variant C", "content": variant_c})
        if variant_d:
            variants_to_test.append({"name": "Variant D", "content": variant_d})
        if variant_e:
            variants_to_test.append({"name": "Variant E", "content": variant_e})
    else:
        if uploaded_file and 'df_upload' in locals():
            for _, row in df_upload.iterrows():
                variants_to_test.append({
                    "name": row['variant_name'],
                    "content": row['content']
                })
    
    if len(variants_to_test) < 2:
        st.error("⚠️ Please enter at least 2 variants to test!")
    else:
        with st.spinner("🧪 Running A/B test..."):
            results = []
            
            for variant in variants_to_test:
                prediction = predictor.predict_engagement(variant['content'], platform)
                
                if prediction:
                    results.append({
                        'Variant': variant['name'],
                        'Content': variant['content'][:100] + '...' if len(variant['content']) > 100 else variant['content'],
                        'Full_Content': variant['content'],
                        'Engagement Score': prediction['predicted_engagement'],
                        'Confidence': prediction['confidence'],
                        'Sentiment': prediction['sentiment']
                    })
            
            if results:
                df_results = pd.DataFrame(results)
                df_results = df_results.sort_values('Engagement Score', ascending=False)
                
                st.session_state.ab_test_results = df_results
                
                # Save to Google Sheets
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                rows_to_save = []
                
                for _, row in df_results.iterrows():
                    rows_to_save.append([
                        timestamp,
                        test_name if test_name else "Unnamed Test",
                        platform,
                        row['Variant'],
                        row['Full_Content'],
                        row['Engagement Score'],
                        row['Confidence'],
                        row['Sentiment']
                    ])
                
                headers = ['Timestamp', 'Test_Name', 'Platform', 'Variant', 'Content', 
                          'Engagement_Score', 'Confidence', 'Sentiment']
                
                if write_to_google_sheets('AB_Test_Tracking', rows_to_save, headers):
                    st.success("✅ Results saved to Google Sheets (AB_Test_Tracking)")

# ==================== DISPLAY RESULTS ====================
if st.session_state.ab_test_results is not None:
    df_results = st.session_state.ab_test_results
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">🏆 Test Results</h3>', unsafe_allow_html=True)
    
    # Winner determination with 3-5% threshold
    winner = df_results.iloc[0]
    runner_up = df_results.iloc[1] if len(df_results) > 1 else None
    
    # Calculate if there's a decisive winner (3-5% threshold)
    has_decisive_winner = False
    gap_percentage = 0
    
    if runner_up is not None:
        engagement_gap = winner['Engagement Score'] - runner_up['Engagement Score']
        gap_percentage = (engagement_gap / max(runner_up['Engagement Score'], 1)) * 100
        has_decisive_winner = gap_percentage >= 3.0
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if has_decisive_winner:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown(f"""
            ## 🥇 Winner: {winner['Variant']}
            
            **📊 Engagement Score (Relative):** {winner['Engagement Score']:.1f}/100  
            **🎯 Confidence:** {winner['Confidence']}  
            **😊 Sentiment:** {winner['Sentiment']:.2f}
            
            **📝 Content:**
            {winner['Full_Content']}
            """)
            
            engagement_gap = winner['Engagement Score'] - runner_up['Engagement Score']
            
            st.markdown(f"""
            ---
            **🔍 Performance Gap:**
            - **{engagement_gap:.1f} points** ahead of {runner_up['Variant']}
            - **{gap_percentage:.1f}% better** relative performance
            - ✅ **Decisive winner** (≥3% difference threshold met)
            """)
        else:
            st.markdown('<div class="warning-box" style="background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%); padding: 20px; border-radius: 10px; color: white;">', unsafe_allow_html=True)
            st.markdown(f"""
            ## ⚖️ Statistically Similar Performance
            
            **Top Variant:** {winner['Variant']}  
            **📊 Engagement Score:** {winner['Engagement Score']:.1f}/100  
            **📉 Performance Gap:** {gap_percentage:.1f}% (below 3% threshold)
            
            **📝 Content:**
            {winner['Full_Content']}
            
            ---
            **💡 Insight:** Variants perform similarly - no decisive winner.  
            Both approaches are equally viable. Consider testing with different audiences or timing.
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Pie chart of relative engagement distribution
        fig = go.Figure(data=[go.Pie(
            labels=df_results['Variant'],
            values=df_results['Engagement Score'],
            hole=0.4,
            marker_colors=['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
        )])
        fig.update_layout(
            title='Relative Score Distribution',
            template='plotly_dark',
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results table
    st.markdown("### 📋 Detailed Results")
    st.dataframe(
        df_results[['Variant', 'Content', 'Engagement Score', 'Confidence', 'Sentiment']],
        use_container_width=True,
        hide_index=True
    )
    
    # Bar chart comparison
    fig = px.bar(
        df_results,
        x='Variant',
        y='Engagement Score',
        color='Engagement Score',
        color_continuous_scale='Viridis',
        title='Relative Engagement Score Comparison',
        text='Engagement Score'
    )
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(
        template='plotly_dark',
        height=400,
        yaxis_title='Relative Engagement Score (0-100)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment comparison
    st.markdown("### 😊 Sentiment Analysis")
    
    fig = px.scatter(
        df_results,
        x='Sentiment',
        y='Engagement Score',
        size='Engagement Score',
        color='Variant',
        title='Sentiment vs Relative Engagement Score',
        labels={'Sentiment': 'Sentiment Score', 'Engagement Score': 'Engagement Score'}
    )
    fig.update_layout(
        template='plotly_dark',
        height=400,
        yaxis_title='Relative Engagement Score (0-100)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📢 Send Results to Slack", key="send_results_slack"):
            winner_status = "🥇 Decisive Winner" if has_decisive_winner else "⚖️ Statistically Similar"
            gap_text = f" ({gap_percentage:.1f}% ahead)" if has_decisive_winner else " (below 3% threshold)"
            
            message = f"""🧪 **A/B Test Results**

📋 **Test Name:** {test_name if test_name else "Unnamed Test"}
🎯 **Platform:** {platform}
📊 **Variants Tested:** {len(df_results)}

{winner_status}: {winner['Variant']}{gap_text}
📈 **Engagement Score (Relative):** {winner['Engagement Score']:.1f}/100
🎯 **Confidence:** {winner['Confidence']}

📝 **Top Content:**
{winner['Full_Content']}

💡 **Note:** Scores are relative (0-100) for comparison, not real engagement predictions.
✅ Results saved to Google Sheets
"""
            if send_slack_notification(message):
                st.success("✅ Sent to Slack!")
    
    with col2:
        if st.button("🔄 Run New Test", key="new_test"):
            st.session_state.ab_test_results = None
            st.rerun()

# ==================== TIPS ====================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.markdown("""
### 💡 A/B Testing Best Practices

1. **Test One Optimization Dimension**  
   Each variant should change ONE clear element:
   - **Hook Enhanced**: Improve only opening 1-2 lines
   - **Clarity Enhanced**: Simplify language, no emotional change
   - **CTA Enhanced**: Add/improve call-to-action only
   - **Emotion Enhanced**: Increase relatability or pain-point

2. **Understand the Scoring System**  
   - Scores are **relative (0-100)**, not real engagement predictions
   - Use for **ranking variants**, not forecasting metrics
   - 3-5% difference required for "decisive winner" status

3. **Preserve Content Integrity**  
   - Variants must stay within ±20% length of original
   - All product details must be preserved
   - URLs and numbers cannot be modified

4. **Track All Results**  
   All tests are automatically saved to Google Sheets for historical analysis

5. **Consider Platform Differences**  
   What works on Twitter may not work on Reddit - test separately!

6. **Context-Aware System**  
   Scoring uses topic-level statistical signals from your {total_posts:,} collected posts.
""".format(total_posts=1761+2143))
st.markdown('</div>', unsafe_allow_html=True)
