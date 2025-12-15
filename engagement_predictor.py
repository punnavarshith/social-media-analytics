"""
Engagement Coach - Context-Aware Content Analyzer
Platform-aware content scoring using historical context + LLM reasoning
Returns RELATIVE scores (0-100) for comparison, NOT real engagement predictions

⚠️ IMPORTANT: This is NOT a predictive model. It provides comparative analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import re
import time
import google.generativeai as genai
from google_sheet_connect import connect_to_google_sheets, get_sheet
from textblob import TextBlob


class EngagementPredictor:
    """
    Engagement Prediction Coach using Gemini Flash
    Context-aware content scoring with historical insights
    Returns RELATIVE scores (0-100), NOT real engagement predictions
    """
    
    def __init__(self, model='gemini-2.5-flash'):
        """Initialize Gemini-powered predictor"""
        self.gc = connect_to_google_sheets()
        self.spreadsheet = get_sheet(self.gc)
        
        # Initialize Gemini
        self.model_name = model
        try:
            # Get API key from secrets
            api_key = self._get_api_key()
            if api_key:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(model)
                self.gemini_available = True
                print(f"✅ Gemini API configured with {model}")
            else:
                self.gemini_available = False
                print("⚠️ Gemini API key not found")
        except Exception as e:
            self.gemini_available = False
            print(f"⚠️ Error configuring Gemini: {e}")
        
        # Historical data cache for context
        self.historical_stats = None
        self.stats_path = 'historical_stats.json'
        self._rate_limit_delay = 0.5
        self._last_request_time = 0
    
    def _get_api_key(self):
        """Get Gemini API key from environment or secrets"""
        # Try environment variable first
        api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
        if api_key:
            return api_key
        
        # Try streamlit secrets
        try:
            import streamlit as st
            api_key = st.secrets.get('google_api_key')
            if api_key:
                return api_key
        except:
            pass
        
        # Try secrets.toml directly
        try:
            try:
                import tomli
            except ImportError:
                # tomli not installed, skip
                return None
                
            secrets_path = os.path.join('.streamlit', 'secrets.toml')
            if os.path.exists(secrets_path):
                with open(secrets_path, 'rb') as f:
                    secrets = tomli.load(f)
                    return secrets.get('google_api_key')
        except:
            pass
        
        return None
        
    def load_historical_data(self):
        """
        Load historical data from Google Sheets for context building
        
        Returns:
            pd.DataFrame: Combined historical data
        """
        print("\n📊 Loading historical data from Google Sheets...")
        
        all_data = []
        
        # Load Twitter data
        try:
            twitter_sheet = self.spreadsheet.worksheet('twitter_data')
            twitter_data = twitter_sheet.get_all_values()
            
            if len(twitter_data) > 1:
                df = pd.DataFrame(twitter_data[1:], columns=twitter_data[0])
                
                # Calculate engagement
                df['likes'] = pd.to_numeric(df['likes'], errors='coerce').fillna(0)
                df['retweets'] = pd.to_numeric(df['retweets'], errors='coerce').fillna(0)
                df['replies'] = pd.to_numeric(df['replies'], errors='coerce').fillna(0)
                df['engagement'] = df['likes'] + df['retweets'] + df['replies']
                
                df['platform'] = 'twitter'
                df['content'] = df['text'] if 'text' in df.columns else ''
                
                all_data.append(df[['content', 'created_at', 'engagement', 'platform']])
                print(f"   ✅ Loaded {len(df)} Twitter posts")
        except Exception as e:
            print(f"   ⚠️ Could not load Twitter data: {e}")
        
        # Load Reddit data
        try:
            reddit_sheet = self.spreadsheet.worksheet('reddit_data')
            reddit_data = reddit_sheet.get_all_values()
            
            if len(reddit_data) > 1:
                df = pd.DataFrame(reddit_data[1:], columns=reddit_data[0])
                
                # Calculate engagement
                df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0)
                df['num_comments'] = pd.to_numeric(df['num_comments'], errors='coerce').fillna(0)
                df['engagement'] = df['score'] + df['num_comments']
                
                df['platform'] = 'reddit'
                df['content'] = df['title'] if 'title' in df.columns else ''
                
                all_data.append(df[['content', 'created_at', 'engagement', 'platform']])
                print(f"   ✅ Loaded {len(df)} Reddit posts")
        except Exception as e:
            print(f"   ⚠️ Could not load Reddit data: {e}")
        
        if not all_data:
            print("   ❌ No historical data available")
            return None
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n✅ Total historical samples: {len(combined_df)}")
        
        return combined_df
    
    def compute_historical_stats(self, df=None):
        """
        Compute statistics from historical data for LLM context
        
        Args:
            df: Historical DataFrame (optional, will load if not provided)
            
        Returns:
            dict: Historical statistics
        """
        if df is None:
            df = self.load_historical_data()
            if df is None:
                return self._get_default_stats()
        
        print("\n📊 Computing historical statistics...")
        
        # Parse timestamps
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['hour'] = df['created_at'].dt.hour
        df['day_of_week'] = df['created_at'].dt.day_name()
        
        # Compute stats
        stats = {
            'total_posts': len(df),
            'platforms': {},
            'overall': {
                'avg_engagement': float(df['engagement'].mean()),
                'median_engagement': float(df['engagement'].median()),
                'max_engagement': float(df['engagement'].max()),
                'std_engagement': float(df['engagement'].std())
            },
            'timing': {
                'best_hours': df.groupby('hour')['engagement'].mean().nlargest(5).to_dict(),
                'best_days': df.groupby('day_of_week')['engagement'].mean().nlargest(3).to_dict()
            },
            'content_patterns': {
                'avg_length': float(df['content'].str.len().mean()),
                'hashtag_impact': self._analyze_hashtag_impact(df),
                'emoji_impact': self._analyze_emoji_impact(df),
                'url_impact': self._analyze_url_impact(df)
            }
        }
        
        # Platform-specific stats
        for platform in df['platform'].unique():
            platform_df = df[df['platform'] == platform]
            stats['platforms'][platform] = {
                'count': len(platform_df),
                'avg_engagement': float(platform_df['engagement'].mean()),
                'median_engagement': float(platform_df['engagement'].median()),
                'top_performing': platform_df.nlargest(5, 'engagement')['engagement'].tolist()
            }
        
        # Save stats
        with open(self.stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✅ Historical stats computed and saved to {self.stats_path}")
        
        return stats
    
    def _analyze_hashtag_impact(self, df):
        """Analyze impact of hashtags on engagement"""
        try:
            df['has_hashtag'] = df['content'].str.contains('#', na=False)
            with_hashtag = df[df['has_hashtag']]['engagement'].mean()
            #df['has_hashtag'] filters rows → only True ones
            #Then take engagement column
            #Then .mean() → average engagement score
            without_hashtag = df[~df['has_hashtag']]['engagement'].mean()
            return {
                'with_hashtag': float(with_hashtag) if pd.notna(with_hashtag) else 0,
                'without_hashtag': float(without_hashtag) if pd.notna(without_hashtag) else 0,
                'improvement': float((with_hashtag / without_hashtag - 1) * 100) if without_hashtag > 0 else 0
            }
        except:
            return {'with_hashtag': 0, 'without_hashtag': 0, 'improvement': 0}
    
    def _analyze_emoji_impact(self, df):
        """Analyze impact of emojis on engagement"""
        try:
            #These are Unicode blocks where emojis are stored.
            df['has_emoji'] = df['content'].str.contains('[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]', regex=True, na=False)
            with_emoji = df[df['has_emoji']]['engagement'].mean()
            without_emoji = df[~df['has_emoji']]['engagement'].mean()
            return {
                'with_emoji': float(with_emoji) if pd.notna(with_emoji) else 0,
                'without_emoji': float(without_emoji) if pd.notna(without_emoji) else 0,
                'improvement': float((with_emoji / without_emoji - 1) * 100) if without_emoji > 0 else 0
            }
        except:
            return {'with_emoji': 0, 'without_emoji': 0, 'improvement': 0}
    
    def _analyze_url_impact(self, df):
        """Analyze impact of URLs on engagement"""
        try:
            df['has_url'] = df['content'].str.contains('http', na=False)
            with_url = df[df['has_url']]['engagement'].mean()
            without_url = df[~df['has_url']]['engagement'].mean()
            return {
                'with_url': float(with_url) if pd.notna(with_url) else 0,
                'without_url': float(without_url) if pd.notna(without_url) else 0,
                'improvement': float((with_url / without_url - 1) * 100) if without_url > 0 else 0
            }
        except:
            return {'with_url': 0, 'without_url': 0, 'improvement': 0}
    
    def _get_default_stats(self):
        """Default stats when no historical data available"""
        return {
            'total_posts': 0,
            'platforms': {},
            'overall': {
                'avg_engagement': 10,
                'median_engagement': 5,
                'max_engagement': 100,
                'std_engagement': 15
            },
            'timing': {
                'best_hours': {9: 15, 12: 14, 18: 13, 6: 12, 15: 11},
                'best_days': {'Monday': 15, 'Wednesday': 14, 'Friday': 13}
            },
            'content_patterns': {
                'avg_length': 150,
                'hashtag_impact': {'improvement': 20},
                'emoji_impact': {'improvement': 15},
                'url_impact': {'improvement': -5}
            }
        }
    
    def _get_sentiment_label(self, sentiment_polarity):
        """
        Convert sentiment polarity to label
        
        Args:
            sentiment_polarity: Float between -1.0 and 1.0
            
        Returns:
            str: 'Positive', 'Neutral', or 'Negative'
        """
        if sentiment_polarity > 0.2:
            return 'Positive'
        elif sentiment_polarity < -0.2:
            return 'Negative'
        else:
            return 'Neutral'
    
    def _normalize_to_relative_score(self, raw_engagement, platform):
        """
        Normalize raw engagement prediction to 0-100 relative score
        
        Args:
            raw_engagement: Raw number from Gemini (likes+comments+shares estimate)
            platform: 'twitter' or 'reddit'
            
        Returns:
            int: Normalized score 0-100
        """
        # Use percentile-based normalization based on historical data
        if self.historical_stats and platform in self.historical_stats.get('platforms', {}):
            platform_stats = self.historical_stats['platforms'][platform]
            avg = platform_stats.get('avg_engagement', 50)
            max_seen = platform_stats.get('max_engagement', 100)
            
            # Calculate percentile
            if raw_engagement <= 0:
                return 0
            elif raw_engagement >= max_seen:
                return 100
            elif raw_engagement <= avg:
                # Below average: scale 0-50
                return int((raw_engagement / avg) * 50)
            else:
                # Above average: scale 50-100
                return int(50 + ((raw_engagement - avg) / (max_seen - avg)) * 50)
        else:
            # Fallback: simple scaling (assume reasonable ranges)
            # Twitter: 0-1000 engagement range
            # Reddit: 0-500 engagement range
            max_range = 1000 if platform == 'twitter' else 500
            normalized = min(100, max(0, int((raw_engagement / max_range) * 100)))
            return normalized
    
    def _calculate_content_quality(self, content, platform):
        """
        Calculate content quality score (1-5 scale)
        
        Args:
            content: Text content
            platform: Platform name
            
        Returns:
            int: Quality score 1-5
        """
        score = 3  # Start with baseline
        
        # Length appropriateness
        length = len(content)
        if platform == 'twitter':
            if 80 <= length <= 220:
                score += 1
            elif length > 280:
                score -= 1
        else:  # reddit
            if length >= 50:
                score += 1
            if length < 20:
                score -= 1
        
        # Readability
        if '?' in content or '!' in content:
            score += 0.5
        
        # Platform fit
        if platform == 'twitter':
            if '#' in content or any(emoji in content for emoji in ['😀','😊','🎉','💪','🔥','✨','🚀']):
                score += 0.5
        else:  # reddit
            if not ('#' in content):  # Reddit doesn't use hashtags
                score += 0.5
        
        # Clamp to 1-5
        return min(5, max(1, int(round(score))))
    
    def _calculate_improvement_potential(self, engagement_score, content, platform):
        """
        Calculate realistic improvement potential
        
        Args:
            engagement_score: Current score (0-100)
            content: Text content
            platform: Platform name
            
        Returns:
            float: Improvement percentage (0-50%)
        """
        # Count issues
        issues = 0
        
        # Platform-specific checks
        if platform == 'twitter':
            if len(content) > 280:
                issues += 1
            if '#' not in content:
                issues += 1
        else:  # reddit
            if '?' not in content and len(content) < 100:
                issues += 1
            if '#' in content:  # hashtags don't work on reddit
                issues += 1
        
        # Realistic improvement caps
        if engagement_score >= 70:
            return 0.0  # Already excellent
        elif engagement_score >= 60:
            return min(25.0, 5.0 + issues * 5.0)  # Good content: max 25%
        elif issues == 0:
            return 5.0
        elif issues == 1:
            return 15.0
        elif issues == 2:
            return 25.0
        else:
            return 40.0  # Max 40% for 3+ issues
    
    def load_model(self):
        """Load historical stats for context building"""
        # Provides historical context for comparative analysis
        if os.path.exists(self.stats_path):
            with open(self.stats_path, 'r') as f:
                self.historical_stats = json.load(f)
            print("✅ Historical stats loaded successfully")
            return True
        else:
            print("⚠️ No historical stats found. Run analyze_historical_data() first.")
            return False
    
    def predict_engagement(self, content, platform='twitter', post_time=None):
        """
        Analyze content and provide relative engagement score (0-100)
        
        Args:
            content: Post text
            platform: 'twitter' or 'reddit'
            post_time: datetime object (default: now)
            
        Returns:
            dict: Analysis results with normalized scores
        """
        # Ensure historical stats are loaded
        if self.historical_stats is None:
            if not self.load_model():
                self.historical_stats = self._get_default_stats()
        
        if not self.gemini_available:
            print("⚠️ Gemini unavailable, using heuristic fallback")
            return self._get_fallback_prediction(content, platform, post_time or datetime.now(), 0.0)
        
        if post_time is None:
            post_time = datetime.now()
        
        # Calculate sentiment
        try:
            blob = TextBlob(content)
            sentiment = blob.sentiment.polarity
        except:
            sentiment = 0.0
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - time_since_last)
        
        # Build context
        hour = post_time.hour
        day_name = post_time.strftime('%A')
        
        # Platform-specific guidance
        platform_rules = {
            'twitter': """
- Emojis: Optional (1-2 acceptable)
- Hashtags: 1-2 relevant tags
- Length: 80-220 chars optimal
- CTAs: Encouraged
""",
            'reddit': """
- Emojis: DO NOT recommend
- Hashtags: DO NOT recommend
- Questions: Valid CTAs
- Length: 50+ words with context
- Discussion-oriented tone
"""
        }
        
        # Create Gemini prompt (concise to avoid token limit)
        prompt = f"""Predict engagement for this social media post.

CONTENT: {content}
PLATFORM: {platform}
TIME: {day_name} {hour}:00
SENTIMENT: {sentiment:.2f}

RULES:
{platform_rules.get(platform, platform_rules['twitter'])}

CRITICAL: predicted_engagement MUST be a NUMBER (not a word like "Low" or "High").
Estimate total engagement as: likes + comments + shares.

Example valid responses:
{{"predicted_engagement": 150, "confidence": "High", "reasoning": "Strong hook"}}
{{"predicted_engagement": 25, "confidence": "Low", "reasoning": "Weak CTA"}}

Output ONLY JSON (no markdown, no extra text):"""

        try:
            # Call Gemini API
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.3,
                    'max_output_tokens': 2048,  # Increased significantly
                    'candidate_count': 1,
                },
                safety_settings=[
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE",
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE",
                    },
                ]
            )
            
            self._last_request_time = time.time()
            
            # Check if response was blocked or incomplete
            if not response or not response.text:
                # Check for finish reason
                finish_reason = getattr(response.candidates[0], 'finish_reason', 'UNKNOWN') if hasattr(response, 'candidates') and len(response.candidates) > 0 else 'UNKNOWN'
                raise Exception(f"Empty or blocked response from Gemini API. Finish reason: {finish_reason}")
            
            # Extract and parse JSON response with multiple strategies
            response_text = response.text.strip()
            result = None
            json_str = None
            
            # Strategy 1: Try markdown code block extraction
            try:
                if '```json' in response_text:
                    json_str = response_text.split('```json')[1].split('```')[0].strip()
                elif '```' in response_text:
                    json_str = response_text.split('```')[1].split('```')[0].strip()
                else:
                    json_str = response_text
                
                result = json.loads(json_str)
            except json.JSONDecodeError:
                pass
            
            # Strategy 2: Try fixing double-quoted strings and common typos
            if result is None and json_str:
                try:
                    import re
                    # Fix double quotes in string values like ""Medium""
                    fixed_json = re.sub(r'"\s*"([^"]+)"\s*"', r'"\1"', json_str)
                    # Fix common typos in keys
                    fixed_json = fixed_json.replace('"confidee nce"', '"confidence"')
                    fixed_json = fixed_json.replace('"predicte d_engagement"', '"predicted_engagement"')
                    fixed_json = fixed_json.replace('"reasoni ng"', '"reasoning"')
                    # Try to complete incomplete JSON by adding missing closing brace and quote
                    if not fixed_json.endswith('}'):
                        # Count braces to see if we need to close
                        open_braces = fixed_json.count('{')
                        close_braces = fixed_json.count('}')
                        if open_braces > close_braces:
                            # Check if reasoning field is incomplete
                            if '"reasoning":' in fixed_json and not fixed_json.rstrip().endswith('"'):
                                fixed_json += '"}'
                            elif fixed_json.rstrip().endswith(':'):
                                fixed_json += ' "incomplete"}'
                            else:
                                fixed_json += '}'
                    result = json.loads(fixed_json)
                except (json.JSONDecodeError, AttributeError) as e:
                    pass
            
            # Strategy 3: Try regex to find JSON object
            if result is None:
                try:
                    import re
                    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        # Also try fixing double quotes here
                        fixed_json = re.sub(r'"\s*"([^"]+)"\s*"', r'"\1"', json_str)
                        result = json.loads(fixed_json)
                except (json.JSONDecodeError, AttributeError):
                    pass
            
            # Strategy 4: Try extracting between first { and last }
            if result is None:
                try:
                    first_brace = response_text.find('{')
                    last_brace = response_text.rfind('}')
                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                        json_str = response_text[first_brace:last_brace+1]
                        # Also try fixing double quotes here
                        fixed_json = re.sub(r'"\s*"([^"]+)"\s*"', r'"\1"', json_str)
                        result = json.loads(fixed_json)
                except json.JSONDecodeError:
                    pass
            
            # If all strategies failed, raise error
            if result is None:
                raise json.JSONDecodeError(
                    f"All parsing strategies failed. Response: {response_text[:200]}",
                    response_text,
                    0
                )
            
            # Validate and extract predicted_engagement
            if 'predicted_engagement' not in result:
                raise ValueError("Missing predicted_engagement in response")
            
            # Get raw engagement from Gemini - validate it's numeric
            predicted_value = result.get('predicted_engagement', 0)
            
            # Handle case where Gemini returns string instead of number
            if isinstance(predicted_value, str):
                # Try to extract number from string
                import re
                numbers = re.findall(r'\d+\.?\d*', predicted_value)
                if numbers:
                    predicted_value = float(numbers[0])
                else:
                    # If no number found, map confidence-like strings to numbers
                    confidence_map = {
                        'low': 20, 'very low': 10,
                        'moderate': 50, 'medium': 50, 'average': 50,
                        'high': 80, 'very high': 90
                    }
                    predicted_value = confidence_map.get(predicted_value.lower(), 50)
            
            raw_engagement = max(0, float(predicted_value))
            
            # Normalize to 0-100 relative score
            engagement_score = self._normalize_to_relative_score(raw_engagement, platform)
            
            # Get sentiment label
            sentiment_label = self._get_sentiment_label(sentiment)
            
            # Calculate content quality
            quality_score = self._calculate_content_quality(content, platform)
            
            # Calculate improvement potential
            improvement_potential = self._calculate_improvement_potential(engagement_score, content, platform)
            
            return {
                'predicted_engagement': engagement_score,  # 0-100 normalized
                'confidence': result.get('confidence', 'Medium'),
                'sentiment': sentiment,  # Raw polarity
                'sentiment_label': sentiment_label,  # UI display
                'quality_score': quality_score,  # 1-5
                'improvement_potential': improvement_potential,  # Capped %
                'reasoning': result.get('reasoning', 'Analysis based on historical patterns and platform best practices')
            }
            
        except Exception as e:
            print(f"⚠️ Gemini API error: {e}, using fallback")
            return self._get_fallback_prediction(content, platform, post_time, sentiment)
    
    def _get_fallback_prediction(self, content, platform, post_time, sentiment):
        """Fallback prediction when LLM fails - uses heuristics"""
        # Simple heuristic-based scoring
        base_score = 50  # Start with average
        
        # Adjust for sentiment
        if sentiment > 0.2:
            base_score += 10
        elif sentiment < -0.2:
            base_score -= 10
        
        # Adjust for platform-specific features
        if platform == 'twitter':
            if '#' in content:
                base_score += 5
            if any(emoji in content for emoji in ['😀','😊','🎉','💪','🔥','✨','🚀']):
                base_score += 5
        else:  # reddit
            if '?' in content:
                base_score += 10
            if len(content) > 100:
                base_score += 5
        
        # Clamp to 0-100
        engagement_score = min(100, max(0, base_score))
        
        # Get other metrics
        sentiment_label = self._get_sentiment_label(sentiment)
        quality_score = self._calculate_content_quality(content, platform)
        improvement_potential = self._calculate_improvement_potential(engagement_score, content, platform)
        
        return {
            'predicted_engagement': engagement_score,
            'confidence': 'Low',
            'sentiment': sentiment,
            'sentiment_label': sentiment_label,
            'quality_score': quality_score,
            'improvement_potential': improvement_potential,
            'reasoning': 'Heuristic-based analysis (LLM unavailable)'
        }
    
    def analyze_historical_data(self):
        """Complete historical analysis pipeline (renamed from train_from_scratch)"""
        print("=" * 80)
        print("📊 HISTORICAL DATA ANALYSIS - Context Building")
        print("=" * 80)
        
        # Load data
        df = self.load_historical_data()
        if df is None or len(df) < 10:
            print("❌ Insufficient historical data (need at least 10 samples)")
            return False
        
        # Compute stats
        stats = self.compute_historical_stats(df)
        self.historical_stats = stats
        
        print("\n" + "=" * 80)
        print("✅ Historical Analysis Complete!")
        print("=" * 80)
        print(f"\n📊 Analyzed {stats['total_posts']} posts")
        print(f"📊 Average engagement: {stats['overall']['avg_engagement']:.1f}")
        print(f"📊 You can now use predict_engagement() for context-aware scoring")
        
        return True
    
    def train_from_scratch(self):
        """Deprecated: Use analyze_historical_data() instead"""
        print("⚠️ train_from_scratch() is deprecated. Use analyze_historical_data() instead.")
        return self.analyze_historical_data()


def main():
    """Test the context-aware predictor"""
    print("=" * 80)
    print("🤖 TESTING ENGAGEMENT COACH - Context-Aware Analyzer")
    print("=" * 80)
    
    predictor = EngagementPredictor()
    
    # Analyze historical data
    success = predictor.analyze_historical_data()
    
    if success:
        # Test prediction
        print("\n" + "=" * 80)
        print("🧪 TESTING CONTENT ANALYSIS")
        print("=" * 80)
        
        test_content = "Exciting news! 🚀 Our new AI-powered analytics platform is now live. Check it out and boost your marketing ROI today! #AI #Marketing #Analytics"
        
        result = predictor.predict_engagement(test_content, platform='twitter')
        
        if result:
            print(f"\n📊 Analysis Results:")
            print(f"   Content: {test_content[:100]}...")
            print(f"   Engagement Score: {result['predicted_engagement']}/100 (Relative)")
            print(f"   Confidence: {result['confidence']}")
            print(f"   Sentiment: {result.get('sentiment_label', 'Neutral')} ({result['sentiment']:.2f})")
            print(f"   Quality Score: {result.get('quality_score', 0)}/5")
            print(f"   Improvement Potential: +{result.get('improvement_potential', 0):.1f}%")
            print(f"   Reasoning: {result['reasoning']}")


if __name__ == "__main__":
    main()
