"""
Engagement Prediction Coach
Platform-aware AI advisor providing actionable, realistic recommendations

🎯 CRITICAL RULES:
1. PLATFORM AWARENESS - Adapt all advice to platform norms (Reddit ≠ Twitter)
2. NORMALIZED SCORING - Use 0-100 relative scores, NOT raw counts
3. REALISTIC IMPROVEMENTS - Low to moderate (5-25%) for good content
4. PLATFORM-SPECIFIC QUALITY - No emoji/hashtag penalties on Reddit
5. ACTIONABLE ADVICE - Specific, relevant recommendations only
6. INTERNAL CONSISTENCY - Metrics must align logically
7. TRANSPARENCY - Explain reasoning in human terms

OUTPUT STRUCTURE:
- Engagement Score (0-100 relative index)
- Confidence Level + explanation
- Platform-specific sentiment analysis
- Platform-aware insights
- Actionable recommendations
- Realistic improvement potential
- Best posting time (heuristic-based)
"""

import pandas as pd
from datetime import datetime, timedelta
from engagement_predictor import EngagementPredictor
from campaign_simulator import CampaignSimulator
from google_sheet_connect import connect_to_google_sheets, get_sheet
from slack_notify import send_slack_message
import json


class PredictionCoach:
    """AI coach for campaign optimization and recommendations"""
    
    def __init__(self):
        """Initialize prediction coach"""
        self.predictor = EngagementPredictor()
        self.simulator = CampaignSimulator()
        self.gc = connect_to_google_sheets()
        self.spreadsheet = get_sheet(self.gc)
        
        # Load model
        self.predictor.load_model()
        
    def analyze_content(self, content, platform='twitter'):
        """
        Comprehensive content analysis with recommendations        
        Args:
            content: Post text to analyze
            platform: Target platform
            
        Returns:
            dict: Analysis and recommendations
        """
        print(f"\n🔍 Analyzing content for {platform}...")
        
        # Get prediction
        prediction = self.predictor.predict_engagement(content, platform)
        
        if not prediction:
            return None
        
        # Get optimal posting time from simulator
        timing_simulation = self.simulator.simulate_timing_variants(content, platform, days=7)
        optimal_time_info = timing_simulation['optimal_time'] if timing_simulation else None
        
        # Analyze content characteristics
        analysis = {
            'content': content,
            'platform': platform,
            'predicted_engagement': prediction['predicted_engagement'],
            'confidence': prediction['confidence'],
            'sentiment': prediction['sentiment'],
            'content_length': len(content),
            'word_count': len(content.split()),
            'has_hashtags': '#' in content,
            'has_emojis': any(emoji in content for emoji in ['😀','😊','🎉','💪','🔥','✨']),
            'has_call_to_action': any(word in content.lower() for word in ['try', 'check', 'click', 'join', 'subscribe', 'buy']),
            'optimal_posting_time': optimal_time_info
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(analysis)
        analysis['recommendations'] = recommendations
        
        # Calculate improvement potential
        analysis['improvement_potential'] = self._calculate_improvement_potential(analysis)
        
        return analysis
    
    def _generate_recommendations(self, analysis):
        """
        Generate platform-specific actionable recommendations
        
        CRITICAL: Recommendations must respect platform norms
        - Reddit: NO emoji/hashtag advice, focus on discussion quality
        - Twitter/Instagram: Emojis/hashtags OK
        """
        recommendations = []
        platform = analysis['platform']
        
        # Platform-specific quality checks
        content = analysis['content'].lower()
        has_question = '?' in content
        has_hook = any(word in content for word in ['why', 'how', 'what', 'when', 'imagine', 'ever wonder'])
        has_examples = any(word in content for word in ['example', 'like', 'such as', 'for instance'])
        has_numbers = any(char.isdigit() for char in content)
        
        # Sentiment recommendations (universal) - REALISTIC classification
        # Reserve "negative" only for clearly hostile/angry content
        if analysis['sentiment'] < -0.5:  # Raised threshold for negative
            recommendations.append("⚠️ Strongly negative sentiment detected. Consider reframing with constructive language to encourage engagement.")
        elif analysis['sentiment'] > 0.5:
            recommendations.append("✅ Excellent positive sentiment! This tone resonates well.")
        # Reflective or complaint-based content (-0.5 to 0.3) = Neutral (no recommendation needed)
        
        # PLATFORM-SPECIFIC RECOMMENDATIONS
        if platform == 'reddit':
            # Reddit-specific advice
            if not has_question and not has_hook:
                recommendations.append("💡 Add a thought-provoking question or strong hook to invite discussion (e.g., 'Has anyone experienced...?', 'What are your thoughts on...')")
            
            if analysis['content_length'] < 50:
                recommendations.append("📝 Reddit posts perform better with more context. Add specific details, examples, or background.")
            
            if not has_examples and not has_numbers:
                recommendations.append("🔢 Add concrete examples or data points to strengthen credibility.")
            
            if analysis['word_count'] > 500:
                recommendations.append("📋 Consider breaking long text into paragraphs with clear structure for better readability.")
            
            # Reddit DOES NOT need hashtags/emojis
            # Questions ARE valid CTAs on Reddit
            
        else:  # Twitter/Instagram
            # Twitter-specific advice
            if analysis['content_length'] > 220:
                recommendations.append("✂️ Twitter performs best under 220 characters. Try condensing to key points.")
            elif analysis['content_length'] < 80:
                recommendations.append("💬 Add more context or a compelling hook to increase impact.")
            
            # Hashtag recommendations (Twitter only)
            if not analysis['has_hashtags']:
                recommendations.append("📌 Add 2-3 relevant hashtags to increase discoverability (e.g., #Marketing #SocialMedia).")
            
            # Emoji recommendations (Twitter only)
            if not analysis['has_emojis']:
                recommendations.append("😊 Add 1-2 relevant emojis to make content more eye-catching.")
            
            # CTA recommendations (Twitter explicit CTAs)
            if not analysis['has_call_to_action'] and not has_question:
                recommendations.append("🎯 Add a clear call-to-action (e.g., 'Learn more', 'Check it out', 'What do you think?').")
        
        # Engagement-based recommendations (all platforms)
        score = analysis['predicted_engagement']
        if score < 30:
            if platform == 'reddit':
                recommendations.append("⚠️ Low engagement potential. Strengthen your hook, add specificity, or pose a clear question.")
            else:
                recommendations.append("⚠️ Low engagement potential. Try adding visual appeal (emojis), hashtags, or use optimal timing.")
        elif score > 70:
            recommendations.append("🚀 High engagement potential! Content is well-optimized for the platform.")
        
        return recommendations
    
    def _calculate_improvement_potential(self, analysis):
        """Calculate how much engagement could improve"""
        current = analysis['predicted_engagement']
        
        # Estimate potential with all optimizations
        potential_boost = 1.0
        #“If this post is missing something important (like positive sentiment, hashtags, emojis, CTA),
        #  it could perform BETTER if you added those things.”
        if analysis['sentiment'] < 0:
            potential_boost *= 1.3  # Positive sentiment boost
        if not analysis['has_hashtags']:
            potential_boost *= 1.2  # Hashtag boost
        if not analysis['has_emojis']:
            potential_boost *= 1.15  # Emoji boost
        if not analysis['has_call_to_action']:
            potential_boost *= 1.25  # CTA boost
        
        potential = current * potential_boost #potential is the predicted engagement after improvements
        improvement = ((potential - current) / current * 100) if current > 0 else 0
        
        return {
            'current': round(current, 2), #current predicted engagement
            'potential': round(potential, 2), #predicted engagement after improvements
            'improvement_pct': round(improvement, 2) #percentage improvement
        }
    
    def recommend_ab_test(self, content, platform='twitter'):
        """
        Recommend A/B test variants for content
        
        Args:
            content: Original content
            platform: Target platform
            
        Returns:
            dict: Test recommendations
        """
        print(f"\n🧪 Generating A/B test recommendations...")
        
        # Create variants with different optimizations
        variants = []
        
        # Variant 1: Add emojis
        if '🚀' not in content and '✨' not in content:
            variant1 = content.replace('.', ' 🚀.')
            variants.append({'name': 'With Emoji', 'content': variant1, 'change': 'Added emojis'})
        
        # Variant 2: Add hashtags
        if '#' not in content:
            variant2 = content + " #Marketing #Business"
            variants.append({'name': 'With Hashtags', 'content': variant2, 'change': 'Added hashtags'})
        
        # Variant 3: Add CTA "call to action," which is a prompt in marketing designed to 
        # get an immediate response, such as "buy now" or "sign up"
        if not any(word in content.lower() for word in ['check', 'try', 'learn']):
            variant3 = content + " Check it out now!"
            variants.append({'name': 'With CTA', 'content': variant3, 'change': 'Added call-to-action'})
        
        # Simulate all variants
        variant_contents = [v['content'] for v in variants]
        simulation_results = self.simulator.simulate_content_variants(
            content, variant_contents, platform
        )
        
        # Combine results
        test_recommendation = {
            'original': content,
            'platform': platform,
            'variants': []
        }
        #variants is a list of dictionaries, where each dictionary describes one A/B test variation.
        for i, variant in enumerate(variants): #So i is used to match the simulation output.
            #simulation_results = DataFrame containing ML predictions
            sim_result = simulation_results[simulation_results['variant'] == f'Variant {i+1}'].iloc[0] if len(simulation_results) > i+1 else None
            #This returns a DataFrame with predictions for each variant:
            if sim_result is not None:
                test_recommendation['variants'].append({
                    'name': variant['name'],
                    'content': variant['content'],
                    'change': variant['change'],
                    'predicted_engagement': sim_result['predicted_engagement'],
                    'improvement_vs_original': sim_result['predicted_engagement'] - simulation_results.iloc[0]['predicted_engagement']
                })
        
        # Sort by predicted engagement
        test_recommendation['variants'].sort(key=lambda x: x['predicted_engagement'], reverse=True)
        
        return test_recommendation
    
    def recommend_campaign_strategy(self, num_posts, campaign_days=7, platform='twitter'):
        """
        Recommend complete campaign strategy
        
        Args:
            num_posts: Number of posts planned
            campaign_days: Campaign duration
            platform: Target platform
            
        Returns:
            dict: Campaign strategy recommendations
        """
        print(f"\n📋 Generating campaign strategy for {num_posts} posts over {campaign_days} days...")
        
        # Get optimal posting schedule
        optimal_times = self.simulator.optimize_posting_schedule(num_posts, duration_days=campaign_days)
        
        # Calculate post frequency
        posts_per_day = num_posts / campaign_days
        
        # Generate strategy
        strategy = {
            'total_posts': num_posts,
            'duration_days': campaign_days,
            'platform': platform,
            'posts_per_day': round(posts_per_day, 2),
            'optimal_posting_times': [t.strftime('%Y-%m-%d %H:%M') for t in optimal_times[:5]],
            'recommendations': []
        }
        
        # Strategy recommendations
        if posts_per_day > 5:
            strategy['recommendations'].append("⚠️ High posting frequency. Consider reducing to 2-3 posts/day to avoid audience fatigue.")
        elif posts_per_day < 1:
            strategy['recommendations'].append("💡 Low posting frequency. Consider posting at least once daily for consistent engagement.")
        else:
            strategy['recommendations'].append("✅ Good posting frequency for maintaining engagement.")
        
        # Day-of-week strategy - analyze distribution
        #Counts how many recommended posts fall on each weekday
        #Example result: {'Monday': 3, 'Wednesday': 2, 'Friday': 5}
        weekday_distribution = {}
        for t in optimal_times:#Counting how many posts fall on each weekday
            day_name = t.strftime('%A')
            weekday_distribution[day_name] = weekday_distribution.get(day_name, 0) + 1
        
        # Find most scheduled day
        #Finds the day with most scheduled posts
        #If that day has >40% of all posts, it recommends it as optimal
        if weekday_distribution:
            best_day = max(weekday_distribution, key=weekday_distribution.get)
            if weekday_distribution[best_day] > num_posts * 0.4: #If best day frequency is more than 40% of all posts recommended
                strategy['recommendations'].append(f"💡 {best_day} appears to be your optimal day based on LLM predictions.")
        
        # Time-of-day strategy
        morning_slots = [t for t in optimal_times if 6 <= t.hour < 9]
        if len(morning_slots) < num_posts * 0.3:
            strategy['recommendations'].append("🌅 Schedule more posts in early morning (6-9 AM) for optimal reach.")
        
        return strategy
    
    def get_performance_insights(self):
        """
        Analyze historical performance and provide insights
        
        Returns:
            dict: Performance insights
        """
        print(f"\n📊 Analyzing historical performance...")
        
        insights = {
            'best_posting_times': [],
            'best_content_types': [],
            'engagement_trends': {},
            'recommendations': []
        }
        
        try:
            # Load historical data
            twitter_sheet = self.spreadsheet.worksheet('twitter_data')
            twitter_data = twitter_sheet.get_all_values()
            
            if len(twitter_data) > 1:
                df = pd.DataFrame(twitter_data[1:], columns=twitter_data[0])
                
                # Convert engagement
                df['likes'] = pd.to_numeric(df['likes'], errors='coerce').fillna(0)
                df['retweets'] = pd.to_numeric(df['retweets'], errors='coerce').fillna(0)
                df['engagement'] = df['likes'] + df['retweets']
                
                # Analyze best times
                df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
                df['hour'] = df['created_at'].dt.hour
                df['day_of_week'] = df['created_at'].dt.day_name()
                
                # Top hours
                hourly_avg = df.groupby('hour')['engagement'].mean().sort_values(ascending=False).head(3)
                insights['best_posting_times'] = [
                    f"{int(hour)}:00 - {avg:.1f} avg engagement" 
                    for hour, avg in hourly_avg.items()
                ]
                
                # Top days
                daily_avg = df.groupby('day_of_week')['engagement'].mean().sort_values(ascending=False)
                insights['engagement_trends']['best_day'] = daily_avg.index[0] if len(daily_avg) > 0 else 'Unknown'
                insights['engagement_trends']['avg_engagement'] = df['engagement'].mean()
                
                # Recommendations based on data
                if daily_avg.index[0] == 'Friday':
                    insights['recommendations'].append("✅ Your data confirms: Friday is your best day!")
                
                insights['recommendations'].append(f"📊 Your average engagement is {df['engagement'].mean():.1f}. Posts above this are performing well.")
                
        except Exception as e:
            print(f"   ⚠️ Could not analyze historical data: {e}")
            insights['recommendations'].append("💡 Collect more data to get personalized insights.")
        
        return insights
    
    def save_coaching_session(self, session_data):
        """Save coaching session to Google Sheets"""
        try:
            # Get or create worksheet
            try:
                worksheet = self.spreadsheet.worksheet('Coaching_Sessions')
            except:
                worksheet = self.spreadsheet.add_worksheet(
                    title='Coaching_Sessions',
                    rows=1000,
                    cols=10
                )
                headers = [
                    'Timestamp', 'Content_Preview', 'Platform', 'Predicted_Engagement',
                    'Confidence', 'Improvement_Potential', 'Recommendations', 'Session_Type'
                ]
                worksheet.update(range_name='A1', values=[headers])
            
            # Prepare row
            row = [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                session_data.get('content', '')[:100],
                session_data.get('platform', ''),
                session_data.get('predicted_engagement', 0),
                session_data.get('confidence', ''),
                session_data.get('improvement_potential', {}).get('improvement_pct', 0),
                '; '.join(session_data.get('recommendations', [])[:3]),
                session_data.get('session_type', 'Content_Analysis')
            ]
            
            worksheet.append_row(row, value_input_option='USER_ENTERED')
            print(f"\n✅ Saved coaching session to Google Sheets")
            
        except Exception as e:
            print(f"\n⚠️ Could not save session: {e}")
    
    def send_coaching_summary(self, analysis):
        """Send coaching summary to Slack"""
        message = f"""🤖 **Prediction Coach Analysis**

**Content:** {analysis['content'][:100]}...

**Predictions:**
📊 Expected Engagement: {analysis['predicted_engagement']:.1f}
🎯 Confidence: {analysis['confidence']}
😊 Sentiment: {analysis['sentiment']:.2f}

**Improvement Potential:**
📈 Current: {analysis['improvement_potential']['current']}
🚀 Potential: {analysis['improvement_potential']['potential']}
💪 Improvement: +{analysis['improvement_potential']['improvement_pct']:.1f}%

**Top Recommendations:**
{chr(10).join(['• ' + rec for rec in analysis['recommendations'][:3]])}

⏰ {analysis['optimal_posting_time']}
"""
        send_slack_message(message, emoji=":robot_face:")


def main():
    """Run prediction coach demo"""
    print("=" * 80)
    print("🤖 PREDICTION COACH - MILESTONE 4")
    print("=" * 80)
    
    coach = PredictionCoach()
    
    # Example content
    test_content = "Check out our new analytics platform! It helps you track social media performance and optimize your content strategy."
    
    # 1. Content Analysis
    print("\n" + "=" * 80)
    print("1. CONTENT ANALYSIS")
    print("=" * 80)
    analysis = coach.analyze_content(test_content, platform='twitter')
    
    if analysis:
        print(f"\n📊 Analysis Results:")
        print(f"   Predicted Engagement: {analysis['predicted_engagement']:.1f}")
        print(f"   Confidence: {analysis['confidence']}")
        print(f"   Sentiment: {analysis['sentiment']:.2f}")
        print(f"   Improvement Potential: +{analysis['improvement_potential']['improvement_pct']:.1f}%")
        
        print(f"\n💡 Recommendations:")
        for rec in analysis['recommendations']:
            print(f"   {rec}")
        
        # Save session
        coach.save_coaching_session(analysis)
        coach.send_coaching_summary(analysis)
    
    # 2. A/B Test Recommendations
    print("\n" + "=" * 80)
    print("2. A/B TEST RECOMMENDATIONS")
    print("=" * 80)
    ab_test = coach.recommend_ab_test(test_content, platform='twitter')
    
    print(f"\n🧪 Recommended A/B Test Variants:")
    for variant in ab_test['variants']:
        print(f"\n   {variant['name']}:")
        print(f"   Change: {variant['change']}")
        print(f"   Predicted Engagement: {variant['predicted_engagement']:.1f}")
        print(f"   Improvement: +{variant['improvement_vs_original']:.1f}")
    
    # 3. Campaign Strategy
    print("\n" + "=" * 80)
    print("3. CAMPAIGN STRATEGY")
    print("=" * 80)
    strategy = coach.recommend_campaign_strategy(num_posts=10, campaign_days=7, platform='twitter')
    
    print(f"\n📋 Campaign Strategy:")
    print(f"   Posts: {strategy['total_posts']} over {strategy['duration_days']} days")
    print(f"   Frequency: {strategy['posts_per_day']:.1f} posts/day")
    print(f"\n   Top 5 Optimal Posting Times:")
    for time in strategy['optimal_posting_times']:
        print(f"   • {time}")
    print(f"\n   Strategy Recommendations:")
    for rec in strategy['recommendations']:
        print(f"   {rec}")
    
    # 4. Performance Insights
    print("\n" + "=" * 80)
    print("4. PERFORMANCE INSIGHTS")
    print("=" * 80)
    insights = coach.get_performance_insights()
    
    if insights['best_posting_times']:
        print(f"\n⏰ Best Posting Times (from your data):")
        for time in insights['best_posting_times']:
            print(f"   • {time}")
    
    if insights['recommendations']:
        print(f"\n💡 Insights:")
        for rec in insights['recommendations']:
            print(f"   {rec}")
    
    print("\n" + "=" * 80)
    print("✅ Prediction Coach Session Complete!")
    print("=" * 80)
    print("\n📊 Check 'Coaching_Sessions' worksheet in Google Sheets")
    print("📢 Check Slack for detailed recommendations")


if __name__ == "__main__":
    main()
