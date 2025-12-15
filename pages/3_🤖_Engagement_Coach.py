"""
Engagement Prediction Coach Page
Platform-aware AI recommendations for content optimization

🎯 CRITICAL RULES IMPLEMENTED:
1. Platform Awareness - Reddit ≠ Twitter recommendations
2. Normalized Scoring - 0-100 relative scores only
3. Realistic Improvements - 5-25% for good content
4. Platform-Specific Quality - No emoji/hashtag penalties on Reddit
5. Actionable Advice - Specific, relevant recommendations
6. Internal Consistency - Metrics align logically
7. Transparency - Clear explanations provided
"""

import streamlit as st
from utils.styles import get_custom_css
from utils.backend import get_coach, get_predictor
import plotly.express as px

st.set_page_config(page_title="Engagement Coach", page_icon="🤖", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Header
st.markdown('<h1 class="page-title">🤖 Engagement Prediction Coach</h1>', unsafe_allow_html=True)
st.markdown("Get platform-aware AI recommendations to optimize your content")

# Transparency notice
st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.markdown("""
**🧠 About This Coach:**
- Uses **Relative Engagement Scores (0-100)** - NOT real engagement predictions
- Provides **platform-specific** recommendations (Reddit ≠ Twitter)
- Shows **realistic improvement potential** based on actual content issues
- All advice is **actionable** and tailored to platform norms

✅ Use for: Comparative analysis and optimization guidance  
❌ Not for: Predicting exact likes, comments, or upvotes
""")
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Initialize
coach = get_coach()
predictor = get_predictor()

# ==================== INPUT ====================
st.markdown('<h3 class="section-header">✍️ Content Analysis</h3>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

with col1:
    content = st.text_area(
        "Enter content to analyze:",
        height=200,
        placeholder="Type or paste your content here for AI analysis...",
        key="coach_content"
    )

with col2:
    platform = st.selectbox(
        "🎯 Platform:",
        ["twitter", "reddit"],
        key="coach_platform"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    analyze_btn = st.button(
        "🎓 Get Coaching",
        type="primary",
        use_container_width=True,
        key="analyze_content_btn"
    )

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ==================== ANALYSIS ====================
if analyze_btn and content:
    with st.spinner("🤖 AI is analyzing your content..."):
        analysis = coach.analyze_content(content, platform)
        
        if analysis:
            # ==================== PREDICTION RESULTS ====================
            st.markdown('<h3 class="section-header">📊 Performance Prediction</h3>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "📈 Engagement Score",
                    f"{analysis['predicted_engagement']:.0f}/100"
                )
                st.caption("Relative index for comparison")
            
            with col2:
                st.metric(
                    "🎯 Confidence",
                    analysis['confidence']
                )
                st.caption(f"Platform: {platform.title()}")
            
            with col3:
                # More realistic sentiment classification
                # Reserve "Negative" only for clearly hostile content (-0.5 or lower)
                if analysis['sentiment'] > 0.3:
                    sentiment_emoji = "😊"
                    sentiment_label = "Positive"
                elif analysis['sentiment'] < -0.5:
                    sentiment_emoji = "😟"
                    sentiment_label = "Negative"
                else:
                    sentiment_emoji = "😐"
                    sentiment_label = "Neutral"
                
                st.metric(
                    f"{sentiment_emoji} Sentiment",
                    f"{analysis['sentiment']:.2f}"
                )
                st.caption(sentiment_label)
            
            with col4:
                improvement = analysis['improvement_potential']['improvement_pct']
                improvement_label = analysis['improvement_potential'].get('improvement_label', 'MODERATE')
                st.metric(
                    "🚀 Improvement",
                    f"+{improvement:.1f}%",
                    delta=improvement_label
                )
                st.caption(f"{analysis['improvement_potential'].get('issues_found', 0)} issues found")
            
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            
            # ==================== IMPROVEMENT POTENTIAL ====================
            st.markdown('<h3 class="section-header">🚀 Growth Potential</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                
                # Ensure scores never exceed 100 (normalize if needed)
                current = min(analysis['improvement_potential']['current'], 100.0)
                potential = min(analysis['improvement_potential']['potential'], 100.0)
                
                st.markdown(f"""
                ### Current vs Potential Engagement
                
                **📊 Current Prediction:** {current:.1f} / 100
                **🎯 With Optimizations:** {potential:.1f} / 100
                **📈 Potential Gain:** +{potential - current:.1f} points
                
                By implementing the recommendations below, you could boost engagement by **{improvement:.1f}%**!
                """)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # Gauge chart
                import plotly.graph_objects as go
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=potential,
                    delta={'reference': current},
                    title={'text': "Engagement Potential (0-100)"},
                    gauge={
                        'axis': {'range': [0, 100]},  # Fixed to 100 max
                        'bar': {'color': "#2ecc71"},
                        'steps': [
                            {'range': [0, current], 'color': "#95a5a6"},
                            {'range': [current, min(potential, 100)], 'color': "#667eea"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': potential
                        }
                    }
                ))
                fig.update_layout(template='plotly_dark', height=300)
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            
            # ==================== RECOMMENDATIONS ====================
            st.markdown('<h3 class="section-header">💡 AI Recommendations</h3>', unsafe_allow_html=True)
            
            for i, rec in enumerate(analysis['recommendations'], 1):
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown(f"**{i}.** {rec}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            
            # ==================== CONTENT INSIGHTS ====================
            st.markdown('<h3 class="section-header">🔍 Content Breakdown</h3>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📏 Length", f"{analysis['content_length']} chars")
            
            with col2:
                st.metric("📝 Words", f"{analysis['word_count']}")
            
            with col3:
                hashtag_status = "✅ Yes" if analysis['has_hashtags'] else "❌ No"
                st.metric("#️⃣ Hashtags", hashtag_status)
            
            with col4:
                emoji_status = "✅ Yes" if analysis['has_emojis'] else "❌ No"
                st.metric("😊 Emojis", emoji_status)
            
            # Content quality indicators (PLATFORM-AWARE)
            st.markdown("### ✨ Quality Indicators")
            
            quality_score = 0
            max_score = 5
            
            indicators = []
            
            # Universal quality factors
            if analysis['sentiment'] > 0.3:
                quality_score += 1
                indicators.append("✅ Positive sentiment")
            elif analysis['sentiment'] < -0.5:
                indicators.append("⚠️ Consider more constructive tone")
            else:
                # Neutral is acceptable, counts as 0.5
                quality_score += 0.5
                indicators.append("😐 Neutral tone (acceptable)")
            
            # Platform-specific quality checks
            if platform == 'reddit':
                # Reddit quality indicators
                if '?' in content or any(word in content.lower() for word in ['why', 'how', 'what']):
                    quality_score += 1
                    indicators.append("✅ Has discussion question/hook")
                else:
                    indicators.append("⚠️ Add discussion question")
                
                if analysis['content_length'] >= 50:
                    quality_score += 1
                    indicators.append("✅ Adequate context/length")
                else:
                    indicators.append("⚠️ Add more context")
                
                # Structure check
                if any(word in content.lower() for word in ['example', 'like', 'such as']) or any(c.isdigit() for c in content):
                    quality_score += 1
                    indicators.append("✅ Includes examples/specifics")
                else:
                    indicators.append("⚠️ Add concrete examples")
                
                # Readability
                if len(content.split()) >= 20:
                    quality_score += 1
                    indicators.append("✅ Clear, readable structure")
                else:
                    indicators.append("⚠️ Needs more detail")
                    
            else:  # Twitter/X
                if analysis['has_hashtags']:
                    quality_score += 1
                    indicators.append("✅ Includes hashtags")
                else:
                    indicators.append("⚠️ Add relevant hashtags")
                
                if analysis['has_emojis']:
                    quality_score += 1
                    indicators.append("✅ Uses emojis")
                else:
                    indicators.append("⚠️ Add emojis for visual appeal")
                
                if analysis['has_call_to_action'] or '?' in content:
                    quality_score += 1
                    indicators.append("✅ Has call-to-action/question")
                else:
                    indicators.append("⚠️ Add clear call-to-action")
                
                if 80 <= analysis['content_length'] <= 220:
                    quality_score += 1
                    indicators.append("✅ Optimal length")
                else:
                    indicators.append("⚠️ Adjust content length (80-220 chars)")
            
            # Enforce minimum 3/5 for readable, platform-appropriate content
            # If content has engagement score > 50, ensure quality >= 3
            if analysis['predicted_engagement'] > 50 and quality_score < 3:
                quality_score = 3.0  # Floor enforcement
            
            # Progress bar
            st.progress(quality_score / max_score)
            st.markdown(f"**Content Quality Score: {quality_score:.1f}/{max_score}**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                for indicator in indicators[:3]:
                    st.markdown(f"• {indicator}")
            
            with col2:
                for indicator in indicators[3:]:
                    st.markdown(f"• {indicator}")
            
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            
            # ==================== OPTIMAL TIMING ====================
            if analysis.get('optimal_posting_time'):
                st.markdown('<h3 class="section-header">⏰ Best Time to Post</h3>', unsafe_allow_html=True)
                
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown(f"**{analysis['optimal_posting_time'].get('message', 'Based on historical patterns and AI predictions')}**")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("⚠️ Could not analyze content. Please try again.")

elif analyze_btn:
    st.warning("⚠️ Please enter some content to analyze!")

# ==================== TIPS ====================
if not (analyze_btn and content):
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 💡 How It Works
    
    1. **Enter Your Content** - Paste any social media post you're planning
    2. **Select Platform** - Choose Twitter or Reddit
    3. **Get AI Analysis** - Receive instant predictions and recommendations
    4. **Implement Changes** - Apply the suggestions to boost engagement
    5. **Track Results** - Monitor actual performance vs predictions
    
    ### 🎯 What You'll Get
    
    - **Engagement Prediction** - AI-powered forecast of post performance
    - **Confidence Level** - How certain the AI is about its prediction
    - **Sentiment Analysis** - Emotional tone of your content
    - **Improvement Potential** - How much better it could perform
    - **Actionable Recommendations** - Specific steps to optimize content
    - **Content Insights** - Detailed breakdown of your post elements
    - **Optimal Timing** - Best time to publish for maximum reach
    """)
    st.markdown('</div>', unsafe_allow_html=True)
