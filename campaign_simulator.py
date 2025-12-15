"""
Campaign Simulation Engine - Milestone 4
Simulate campaign scenarios and predict outcomes
Tests different content, timing, and platform strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from engagement_predictor import EngagementPredictor
from google_sheet_connect import connect_to_google_sheets, get_sheet
from slack_notify import send_slack_message
import random


class CampaignSimulator:
    """Simulate social media campaigns and predict outcomes"""
    
    def __init__(self):
        """Initialize simulator"""
        self.predictor = EngagementPredictor()
        self.predictor.load_model()
        self.gc = connect_to_google_sheets()
        self.spreadsheet = get_sheet(self.gc)
        
    def simulate_single_post(self, content, platform, post_time):
        """
        Simulate a single post
        
        Args:
            content: Post text
            platform: 'twitter' or 'reddit'
            post_time: datetime object
            
        Returns:
            dict: Simulation results or None if Ollama is unavailable
        """
        try:
            prediction = self.predictor.predict_engagement(content, platform, post_time)
            
            if prediction:
                return {
                    'content': content[:100],
                    'platform': platform,
                    'post_time': post_time.strftime('%Y-%m-%d %H:%M'),
                    'predicted_engagement': prediction['predicted_engagement'],
                    'confidence': prediction['confidence'],
                    'sentiment': round(prediction['sentiment'], 3)
                }
            return None
        except RuntimeError as e:
            print(f"   ⚠️ {e}")
            return None
    
    def simulate_timing_variants(self, content, platform='twitter', days=7):
        """
        Test different posting times for the same content
        
        Args:
            content: Post text
            platform: Platform to simulate
            days: How many days ahead to simulate
            
        Returns:
            dict: Results with optimal time recommendation
        """
        total_tests = days * 6
        print(f"\n⏰ Simulating posting times for next {days} days ({total_tests} predictions)...")
        
        results = []
        base_time = datetime.now()
        test_count = 0
        
        # Test different hours and days
        for day in range(days):
            for hour in [6, 9, 12, 15, 18, 21]:  # Key hours
                test_count += 1
                print(f"   Testing {test_count}/{total_tests}: {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][(base_time.weekday() + day) % 7]} {hour}:00", end='\r')
                test_time = base_time + timedelta(days=day, hours=hour-base_time.hour)
                result = self.simulate_single_post(content, platform, test_time)
                
                if result:
                    result['day_name'] = test_time.strftime('%A')
                    result['hour'] = test_time.hour
                    result['day_offset'] = day
                    results.append(result)
        
        print(f"   ✅ Completed {len(results)} predictions                    ")
        
        df = pd.DataFrame(results).sort_values('predicted_engagement', ascending=False)
        
        # Get optimal time
        if len(df) > 0:
            best = df.iloc[0]
            current_prediction = self.predictor.predict_engagement(content, platform, base_time)
            current_engagement = current_prediction['predicted_engagement'] if current_prediction else 0
            
            improvement = ((best['predicted_engagement'] - current_engagement) / current_engagement * 100) if current_engagement > 0 else 0
            
            optimal_info = {
                'optimal_time': best['post_time'],
                'optimal_day': best['day_name'],
                'optimal_hour': best['hour'],
                'predicted_engagement': best['predicted_engagement'],
                'current_engagement': current_engagement,
                'improvement_pct': round(improvement, 1),
                'message': f"💡 Post on {best['day_name']} at {best['hour']}:00 for +{round(improvement, 1)}% engagement" if improvement > 5 else "✅ Current time is optimal!"
            }
        else:
            optimal_info = None
        
        print(f"\n🏆 Top 3 Posting Times:")
        for idx, row in df.head(3).iterrows():
            print(f"   {row['day_name']} at {row['post_time'].split()[1]}: {row['predicted_engagement']:.1f} engagement")
        
        if optimal_info:
            print(f"\n{optimal_info['message']}")
        
        return {
            'results': df,
            'optimal_time': optimal_info
        }
    
    def simulate_content_variants(self, base_content, variants, platform='twitter', post_time=None):
        """
        Test different content variations
        
        Args:
            base_content: Original content
            variants: List of variant texts
            platform: Platform to test on
            post_time: When to post (default: dynamically calculated optimal time)
            
        Returns:
            pd.DataFrame: Results sorted by predicted engagement
        """
        print(f"\n📝 Simulating {len(variants) + 1} content variants...")
        
        if post_time is None:
            # Find optimal time dynamically using base content
            print("   Finding optimal posting time first...")
            timing_analysis = self.simulate_timing_variants(base_content, platform, days=7)
            if timing_analysis and 'optimal_time' in timing_analysis and timing_analysis['optimal_time']:
                optimal_time_str = timing_analysis['optimal_time']['optimal_time']
                post_time = datetime.strptime(optimal_time_str, '%Y-%m-%d %H:%M')
            else:
                # Fallback to current time if analysis fails
                post_time = datetime.now()
        
        results = []
        
        # Test base content
        print(f"   Testing Original content...")
        result = self.simulate_single_post(base_content, platform, post_time)
        if result:
            result['variant'] = 'Original'
            results.append(result)
        
        # Test variants
        for i, variant in enumerate(variants, 1):
            print(f"   Testing Variant {i}/{len(variants)}...")
            result = self.simulate_single_post(variant, platform, post_time)
            if result:
                result['variant'] = f'Variant {i}'
                results.append(result)
        
        df = pd.DataFrame(results).sort_values('predicted_engagement', ascending=False)
        
        print(f"\n🏆 Best Performing Variant:")
        if len(df) > 0:
            best = df.iloc[0]
            print(f"   {best['variant']}: {best['predicted_engagement']:.1f} engagement")
            print(f"   Content: {best['content']}...")
        
        return df
    
    def simulate_platform_comparison(self, content, post_time=None):
        """
        Compare predicted performance across platforms
        
        Args:
            content: Post text
            post_time: When to post
            
        Returns:
            pd.DataFrame: Platform comparison
        """
        print(f"\n🌐 Comparing platforms...")
        
        if post_time is None:
            post_time = datetime.now()
        
        results = []
        platforms = ['twitter', 'reddit']
        
        for platform in platforms:
            result = self.simulate_single_post(content, platform, post_time)
            if result:
                results.append(result)
        
        df = pd.DataFrame(results).sort_values('predicted_engagement', ascending=False)
        
        print(f"\n🏆 Best Platform:")
        if len(df) > 0:
            best = df.iloc[0]
            print(f"   {best['platform'].title()}: {best['predicted_engagement']:.1f} engagement")
        
        return df
    
    def simulate_full_campaign(self, content_list, start_date=None, duration_days=7, platform='twitter'):
        """
        Simulate a full multi-post campaign
        
        Args:
            content_list: List of posts to schedule
            start_date: Campaign start date
            duration_days: Campaign duration
            platform: Platform to use
            
        Returns:
            dict: Campaign simulation results
        """
        print(f"\n🚀 Simulating {duration_days}-day campaign with {len(content_list)} posts...")
        
        if start_date is None:
            start_date = datetime.now()
        
        results = []
        total_predicted_engagement = 0
        
        # Distribute posts evenly across campaign
        posts_per_day = max(1, len(content_list) // duration_days)
        
        for day in range(duration_days):
            # Post at optimal hours
            optimal_hours = [6, 12, 18]  # Morning, noon, evening
            
            for i, hour in enumerate(optimal_hours[:posts_per_day]):
                post_idx = day * posts_per_day + i #This gives the index of the content to post from the content_list.
                if post_idx >= len(content_list):#If you run out of content, stop posting.
                    break
                
                content = content_list[post_idx] #Take the correct post text
                post_time = start_date + timedelta(days=day, hours=hour)
                
                result = self.simulate_single_post(content, platform, post_time)
                if result:
                    results.append(result)
                    total_predicted_engagement += result['predicted_engagement']
        
        # Calculate campaign metrics
        campaign_stats = {
            'total_posts': len(results),
            'predicted_total_engagement': round(total_predicted_engagement, 2),
            'predicted_avg_engagement': round(total_predicted_engagement / len(results), 2) if results else 0,
            'duration_days': duration_days,
            'platform': platform,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'posts': results
        }
        
        print(f"\n📊 Campaign Simulation Results:")
        print(f"   Posts Scheduled: {campaign_stats['total_posts']}")
        print(f"   Predicted Total Engagement: {campaign_stats['predicted_total_engagement']:.1f}")
        print(f"   Predicted Avg per Post: {campaign_stats['predicted_avg_engagement']:.1f}")
        
        return campaign_stats
    
    def optimize_posting_schedule(self, num_posts, start_date=None, duration_days=7):
        """
        Find optimal posting schedule with TRUE distribution across campaign days
        
        Args:
            num_posts: Number of posts to schedule
            start_date: Start date (default: today)
            duration_days: Campaign duration (default: 7 days)
            
        Returns:
            list: Optimal posting times distributed across the campaign period
        """
        print(f"\n📅 Scheduling {num_posts} posts over {duration_days} days...")
        
        if start_date is None:
            start_date = datetime.now()
        
        # CRITICAL FIX: Distribute posts across days using deterministic logic
        # Formula: day_gap = total_days / total_posts
        #          day_offset = floor(post_index * day_gap)
        
        import math
        
        scheduled_times = []
        day_gap = duration_days / num_posts
        
        # Best hours for engagement (based on typical social media usage)
        optimal_hours = [9, 12, 15, 18, 20]  # Morning, noon, afternoon, evening, night
        
        for i in range(num_posts):
            # Calculate which day this post should be on
            day_offset = math.floor(i * day_gap)
            
            # Ensure day_offset doesn't exceed duration_days - 1
            day_offset = min(day_offset, duration_days - 1)
            
            # Select optimal hour (cycle through best hours)
            hour = optimal_hours[i % len(optimal_hours)]
            
            # Create the scheduled datetime
            post_datetime = start_date + timedelta(days=day_offset, hours=hour)
            
            scheduled_times.append(post_datetime)
            
            print(f"   Post {i+1}: {post_datetime.strftime('%Y-%m-%d %A')} at {hour:02d}:00 (Day {day_offset})")
        
        print(f"\n✅ Schedule created:")
        print(f"   Start: {scheduled_times[0].strftime('%Y-%m-%d')}")
        print(f"   End:   {scheduled_times[-1].strftime('%Y-%m-%d')}")
        print(f"   Span:  {(scheduled_times[-1] - scheduled_times[0]).days} days")
        
        return scheduled_times
    
    def save_simulation_to_sheets(self, simulation_results, simulation_type):
        """Save simulation results to Google Sheets"""
        try:
            # Get or create worksheet
            try:
                worksheet = self.spreadsheet.worksheet('Campaign_Simulations')
            except:
                worksheet = self.spreadsheet.add_worksheet(
                    title='Campaign_Simulations',
                    rows=1000,
                    cols=10
                )
                headers = [
                    'Timestamp', 'Simulation_Type', 'Content_Preview', 'Platform',
                    'Post_Time', 'Predicted_Engagement', 'Confidence', 'Sentiment'
                ]
                worksheet.update(range_name='A1', values=[headers])
            
            # Prepare rows
            rows = []
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            #This block takes a DataFrame of simulation results and converts each row into a list,
            #  so that it can be written to Google Sheets.
            if isinstance(simulation_results, pd.DataFrame):
                for _, row in simulation_results.iterrows():
                    rows.append([
                        timestamp,
                        simulation_type,
                        row.get('content', '')[:100],
                        row.get('platform', ''),
                        row.get('post_time', ''),
                        row.get('predicted_engagement', 0),
                        row.get('confidence', ''),
                        row.get('sentiment', 0)
                    ])
            
            if rows:
                worksheet.append_rows(rows, value_input_option='USER_ENTERED')
                print(f"\n✅ Saved {len(rows)} simulation results to Google Sheets")
            
        except Exception as e:
            print(f"\n⚠️ Could not save to sheets: {e}")
    
    def send_simulation_notification(self, simulation_type, best_result):
        """Send Slack notification with simulation results"""
        message = f"""🔮 **Campaign Simulation Complete**

**Type:** {simulation_type}

**Best Result:**
📝 {best_result.get('content', 'N/A')}...
📊 Predicted Engagement: {best_result.get('predicted_engagement', 0):.1f}
🎯 Confidence: {best_result.get('confidence', 'N/A')}
📅 Optimal Time: {best_result.get('post_time', 'N/A')}
📱 Platform: {best_result.get('platform', 'N/A').title()}

💡 Use this insight to optimize your campaign!
"""
        send_slack_message(message, emoji=":crystal_ball:")


def main():
    """Run campaign simulation examples"""
    print("=" * 80)
    print("🔮 CAMPAIGN SIMULATION ENGINE - MILESTONE 4")
    print("=" * 80)
    
    simulator = CampaignSimulator()
    
    # Example content
    content = "Exciting news! 🚀 Our new AI-powered analytics platform is now live. Check it out and boost your marketing ROI today! #AI #Marketing #Analytics"
    
    # 1. Test timing
    print("\n" + "=" * 80)
    print("1. TIMING OPTIMIZATION")
    print("=" * 80)
    timing_results = simulator.simulate_timing_variants(content, platform='twitter', days=7)
    simulator.save_simulation_to_sheets(timing_results['results'], 'Timing_Optimization')
    
    # 2. Test content variants
    print("\n" + "=" * 80)
    print("2. CONTENT VARIANTS")
    print("=" * 80)
    variants = [
        "New AI analytics platform launched! 🎉 Transform your marketing strategy with real-time insights. #AI #MarketingTech",
        "Boost your ROI with our new analytics platform 📊 AI-powered insights for smarter marketing decisions. Try it now! #Marketing",
        "Revolutionary AI analytics now available! Get actionable insights and drive better results. #Analytics #Business"
    ]
    content_results = simulator.simulate_content_variants(content, variants, platform='twitter')
    simulator.save_simulation_to_sheets(content_results, 'Content_Variants')
    
    # 3. Platform comparison
    print("\n" + "=" * 80)
    print("3. PLATFORM COMPARISON")
    print("=" * 80)
    platform_results = simulator.simulate_platform_comparison(content)
    simulator.save_simulation_to_sheets(platform_results, 'Platform_Comparison')
    
    # Send notification
    if timing_results and 'results' in timing_results:
        results_df = timing_results['results']
        if len(results_df) > 0:
            best_result = results_df.iloc[0].to_dict()
            simulator.send_simulation_notification('Multi-Scenario Test', best_result)
    
    print("\n" + "=" * 80)
    print("✅ Campaign Simulation Complete!")
    print("=" * 80)
    print("\n📊 Check 'Campaign_Simulations' worksheet in Google Sheets")


if __name__ == "__main__":
    main()
