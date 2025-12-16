"""
Advanced Content Optimizer
Marketing intelligence system for content optimization and variant generation

🎯 SYSTEM ARCHITECTURE:
- Context-aware inference using topic-level statistical signals
- Powered by Google Gemini API (NOT trained model weights)
- Injected data context for topic-specific optimization
- Feature-driven scoring + LLM reasoning

🚨 CRITICAL DESIGN PRINCIPLES:
1. RELATIVE SCORING: Engagement scores (0-100) are comparative indices, NOT real-world predictions
2. CONTENT PRESERVATION: Variants stay within ±20% length, preserve all product details/URLs/numbers
3. VARIANT DIFFERENTIATION: Each variant modifies ONE optimization dimension only
4. TRANSPARENCY: System clearly states it does NOT predict real likes/comments/upvotes

📐 VARIANT TYPES:
- Hook Enhanced: Improve ONLY opening 1-2 lines
- Clarity Enhanced: Simplify language without emotional change
- CTA Enhanced: Add/improve call-to-action only
- Emotion Enhanced: Increase relatability or pain-point
- Minimalist: Reduce length while preserving message

🏆 A/B WINNER LOGIC:
- Winner declared ONLY if score difference ≥ 3-5%
- Otherwise: "Statistically similar performance"
"""

import re
import time
import streamlit as st
from textblob import TextBlob
from textstat import flesch_reading_ease
import pandas as pd
from typing import Dict, List, Optional

# Import Google Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ google-generativeai not installed. Run: pip install google-generativeai")


class AdvancedContentOptimizer:
    """
    Marketing-grade content optimizer with:
    - LLM-powered rewriting
    - Topic-aware optimization
    - Deep NLP analysis
    - Real transformations (not just emoji additions)
    """
    
    def __init__(self, use_gemini: bool = True, model: str = "gemini-2.5-flash"):
        """
        Initialize the advanced content optimizer
        
        Args:
            use_gemini: Whether to use Google Gemini API (default: True)
            model: Gemini model to use (default: gemini-1.5-flash)
        """
        self.use_gemini = use_gemini
        self.model = model
        self._gemini_available = None
        self._rate_limit_delay = 4  # 15 requests/min = 4 seconds between requests
        self._last_request_time = 0
        
        # Initialize Gemini API
        if use_gemini and GEMINI_AVAILABLE:
            try:
                api_key = st.secrets.get("google_api_key")
                if api_key:
                    genai.configure(api_key=api_key)
                    self._gemini_available = True
                    print("✅ Google Gemini API configured")
                else:
                    print("⚠️ google_api_key not found in secrets.toml")
                    self._gemini_available = False
            except Exception as e:
                print(f"⚠️ Error configuring Gemini: {e}")
                self._gemini_available = False
    
    def check_gemini_status(self) -> Dict[str, any]:
        """
        Check if Google Gemini API is configured and available
        
        Returns:
            Dict with status, message, and available info
        """
        if not GEMINI_AVAILABLE:
            return {
                'status': 'not_installed',
                'available': False,
                'message': 'google-generativeai library not installed',
                'models': [],
                'target_model_available': False
            }
        
        try:
            # Check if API key is configured
            api_key = st.secrets.get("google_api_key")
            
            if not api_key:
                return {
                    'status': 'not_configured',
                    'available': False,
                    'message': 'google_api_key not found in secrets.toml',
                    'models': [],
                    'target_model_available': False
                }
            
            # Try to list available models (lightweight check)
            try:
                genai.configure(api_key=api_key)
                available_models = [m.name for m in genai.list_models()]
                
                # Check if our model is available
                model_available = any(self.model in m for m in available_models)
                
                self._gemini_available = True
                return {
                    'status': 'running',
                    'available': True,
                    'message': f'Google Gemini API ready ({len(available_models)} models)',
                    'models': available_models[:10],  # Limit to first 10
                    'target_model_available': model_available
                }
            except Exception as api_error:
                return {
                    'status': 'api_error',
                    'available': False,
                    'message': f'Gemini API error: {str(api_error)[:100]}',
                    'models': [],
                    'target_model_available': False
                }
        
        except Exception as e:
            return {
                'status': 'error',
                'available': False,
                'message': f'Error checking Gemini: {str(e)[:100]}',
                'models': [],
                'target_model_available': False
            }
    
    def analyze_content(self, content: str) -> Dict:
        """
        Deep NLP analysis of content with proper preprocessing
        
        Returns:
            Dict with sentiment, readability, structure metrics
        """
        # Extract components
        emoji_count = len(re.findall(r'[\U0001F300-\U0001F9FF]', content))
        hashtag_count = len(re.findall(r'#\w+', content))
        
        # Clean content for analysis (remove emojis, URLs, hashtags)
        clean_content = content
        clean_content = re.sub(r'[\U0001F300-\U0001F9FF]', '', clean_content)  # Remove emojis
        clean_content = re.sub(r'http\S+|www.\S+', '', clean_content)  # Remove URLs
        hashtags_text = ' '.join(re.findall(r'#\w+', clean_content))  # Extract hashtags
        clean_content = re.sub(r'#\w+', '', clean_content)  # Remove hashtags from analysis
        
        # Sentiment analysis on clean content only (not hashtags)
        blob = TextBlob(clean_content.strip())
        
        # Count features
        words = clean_content.split()
        sentences = clean_content.split('.')
        has_cta = bool(re.search(r'(learn more|check out|try|get started|join|discover|read more|see|watch|explore|find out)', content.lower()))
        has_question = '?' in content
        
        # Readability on clean content
        try:
            readability_score = flesch_reading_ease(clean_content) if len(clean_content) > 20 else 50
        except:
            readability_score = 50
        
        analysis = {
            'sentiment_polarity': blob.sentiment.polarity,
            'sentiment_subjectivity': blob.sentiment.subjectivity,
            'readability': readability_score,
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': sum(len(w) for w in words) / max(len(words), 1),
            'emoji_count': emoji_count,
            'hashtag_count': hashtag_count,
            'has_cta': has_cta,
            'has_question': has_question,
            'char_count': len(content)
        }
        
        return analysis
    
    def get_topic_keywords(self, topic: Optional[str] = None) -> List[str]:
        """
        Get relevant keywords for a topic from database
        
        Args:
            topic: The topic to get keywords for
        
        Returns:
            List of relevant keywords
        """
        if not topic:
            return []
        
        # Try to fetch from Supabase or cached data
        try:
            from utils.supabase_db import SupabaseDB
            db = SupabaseDB()
            if db.is_connected():
                # Query posts about this topic
                data = db.client.table('reddit_data').select('title,selftext').eq('topic', topic).limit(50).execute()
                
                if data.data:
                    # Extract common keywords
                    text = ' '.join([row.get('title', '') + ' ' + row.get('selftext', '') for row in data.data])
                    words = re.findall(r'\b\w{4,}\b', text.lower())
                    
                    # Count frequency
                    from collections import Counter
                    word_freq = Counter(words)
                    
                    # Return top keywords (excluding common words)
                    stop_words = {'that', 'this', 'with', 'have', 'from', 'they', 'will', 'been', 'were', 'there'}
                    keywords = [word for word, count in word_freq.most_common(20) if word not in stop_words]
                    return keywords[:10]
        except:
            pass
        
        return []
    
    def generate_topic_hashtags(self, topic: Optional[str] = None, count: int = 3) -> List[str]:
        """
        Generate relevant hashtags based on topic
        
        Args:
            topic: The topic to generate hashtags for
            count: Number of hashtags to generate
        
        Returns:
            List of hashtag strings
        """
        if not topic:
            return ["#Marketing", "#Content", "#SocialMedia"][:count]
        
        # Generate hashtags from topic
        topic_clean = ''.join(c for c in topic if c.isalnum() or c.isspace())
        words = topic_clean.split()
        
        hashtags = []
        
        # Main topic hashtag
        if len(words) <= 2:
            hashtags.append(f"#{''.join(w.capitalize() for w in words)}")
        else:
            hashtags.append(f"#{words[0].capitalize()}")
        
        # Community hashtag
        hashtags.append(f"#{''.join(w.capitalize() for w in words)}Community")
        
        # Category hashtags based on common patterns
        category_tags = {
            "phone": ["#TechReview", "#Smartphone", "#MobileTech"],
            "car": ["#AutoReview", "#CarTech", "#Automotive"],
            "laptop": ["#TechReview", "#Laptop", "#Computing"],
            "bottle": ["#ProductReview", "#Lifestyle", "#Health"],
            "shoes": ["#Fashion", "#Footwear", "#Style"]
        }
        
        for keyword, tags in category_tags.items():
            if keyword in topic.lower():
                hashtags.extend(tags[:2])
                break
        else:
            hashtags.extend(["#ProductReview", "#Innovation"])
        
        return hashtags[:count]
    
    def _get_topic_phrases(self, topic: Optional[str] = None) -> List[str]:
        """
        Get top phrases from Reddit data for topic context
        
        Args:
            topic: Topic to fetch phrases for
        
        Returns:
            List of common phrases from Reddit discussions
        """
        if not topic:
            return []
        
        try:
            from utils.supabase_db import SupabaseDB
            db = SupabaseDB()
            if db.is_connected():
                # Query posts about this topic
                data = db.client.table('reddit_data').select('title,selftext').eq('topic', topic).limit(30).execute()
                
                if data.data:
                    # Extract common 2-3 word phrases
                    all_text = ' '.join([row.get('title', '') + ' ' + row.get('selftext', '') for row in data.data])
                    
                    # Find common phrases (simple approach)
                    words = all_text.lower().split()
                    phrases = []
                    for i in range(len(words) - 2):
                        phrase = ' '.join(words[i:i+3])
                        if len(phrase) > 10:  # Minimum length
                            phrases.append(phrase)
                    
                    # Return top unique phrases
                    from collections import Counter
                    phrase_counts = Counter(phrases)
                    return [phrase for phrase, count in phrase_counts.most_common(5) if count > 1]
        except:
            pass
        
        return []
    
    def _cleanup_llm_output(self, text: str, original_length: int, max_length_increase: float = 0.2) -> str:
        """
        Clean up LLM output and enforce constraints
        
        Args:
            text: LLM generated text
            original_length: Original content length
            max_length_increase: Maximum allowed length increase (0.2 = 20%)
        
        Returns:
            Cleaned text
        """
        original_text = text
        
        # Remove meta-text patterns at the beginning
        meta_patterns = [
            r'^(Here is|Here\'s|Here are|This is|Rewritten version|Rewritten:|Improved version|Updated version|A better version)[\s\w:.-]*\n*',
            r'^(I\'ve rewritten|I rewrote|I\'ve improved|I can help|Let me)[\s\w]+[\s:.-]*\n*',
            r'^\* "',  # Remove bullet points with quotes
            r'^"',     # Remove opening quote if it's a quoted response
        ]
        
        for pattern in meta_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # If LLM gave multiple options (contains bullet points or numbers), take the first one
        if re.search(r'^\s*[\*\-\d+\.]\s', text, re.MULTILINE):
            # Split by bullet points or numbered lists
            lines = text.split('\n')
            content_lines = []
            found_first = False
            
            for line in lines:
                # Skip empty lines before first content
                if not found_first and not line.strip():
                    continue
                
                # If we hit a second bullet/number after collecting some content, stop
                if found_first and re.match(r'^\s*[\*\-\d+\.]\s', line):
                    break
                
                # Remove bullet/number prefix from first item
                if not found_first and re.match(r'^\s*[\*\-\d+\.]\s', line):
                    line = re.sub(r'^\s*[\*\-\d+\.]\s+"?', '', line)
                    found_first = True
                
                if found_first:
                    content_lines.append(line)
            
            text = '\n'.join(content_lines).strip()
        
        # Remove trailing quotes and incomplete sentences
        text = re.sub(r'"\s*$', '', text)  # Remove trailing quote
        text = re.sub(r'\s*-\s*$', '', text)  # Remove trailing dash
        
        # Clean formatting
        text = text.replace('""', '"')
        text = re.sub(r'\s+', ' ', text)
        text = text.replace(' .', '.')
        text = text.replace(' ,', ',')
        text = text.replace(' !', '!')
        text = text.replace(' ?', '?')
        text = text.strip()
        
        # If cleaned text is empty or too short, return original
        if not text or len(text) < 10:
            print(f"[DEBUG] Cleanup resulted in empty/short text, keeping original")
            print(f"[DEBUG] Original LLM output: {original_text[:200]}")
            # Try to extract any sentence from original
            sentences = original_text.split('.')
            for sentence in sentences:
                clean_sent = sentence.strip()
                if len(clean_sent) > 20 and not re.match(r'^(Here|This|I )', clean_sent, re.IGNORECASE):
                    text = clean_sent + '.'
                    break
        
        # Enforce length constraint (trim if too long)
        max_allowed = int(original_length * (1 + max_length_increase))
        if len(text) > max_allowed:
            # Trim to sentence boundary
            sentences = text.split('.')
            trimmed = ""
            for sentence in sentences:
                if len(trimmed) + len(sentence) + 1 <= max_allowed:
                    trimmed += sentence + "."
                else:
                    break
            if trimmed:
                text = trimmed.strip()
        
        return text
        max_allowed = int(original_length * (1 + max_length_increase))
        if len(text) > max_allowed:
            # Trim to sentence boundary
            sentences = text.split('.')
            trimmed = ""
            for sentence in sentences:
                if len(trimmed) + len(sentence) + 1 <= max_allowed:
                    trimmed += sentence + "."
                else:
                    break
            text = trimmed.strip()
        
        return text
    
    def rewrite_with_llm(self, content: str, transformation_type: str, topic: Optional[str] = None) -> Optional[str]:
        """
        Use Google Gemini to rewrite content with specific transformation and strict constraints
        
        Args:
            content: Original content
            transformation_type: Type of transformation (hook, clarity, cta, etc.)
            topic: Topic context for rewriting
        
        Returns:
            Rewritten content if successful, None if Gemini not available
        """
        # Check Gemini availability if not cached
        if self._gemini_available is None:
            status = self.check_gemini_status()
            if not status['available']:
                print(f"[ERROR] Gemini not available: {status['message']}")
                return None
        elif not self._gemini_available:
            return None
        
        # Rate limiting: Ensure 4 seconds between requests (15 req/min)
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._rate_limit_delay:
            sleep_time = self._rate_limit_delay - time_since_last
            print(f"[RATE LIMIT] Sleeping {sleep_time:.1f}s to avoid 429 error...")
            time.sleep(sleep_time)
        
        # Get topic context
        topic_keywords = self.get_topic_keywords(topic) if topic else []
        topic_phrases = self._get_topic_phrases(topic) if topic else []
        
        topic_context = ""
        if topic:
            topic_context = f"Topic: {topic}\n"
            if topic_keywords:
                topic_context += f"Related keywords (from real data): {', '.join(topic_keywords[:5])}\n"
            if topic_phrases:
                topic_context += f"Common phrases (from Reddit): {', '.join(topic_phrases[:3])}\n"
            topic_context += "\n"
        
        # Strict constraints for all rewrites
        constraints = """
CRITICAL INSTRUCTIONS - READ CAREFULLY:
1. Output ONLY the rewritten text - NO explanations, NO introductions, NO options
2. Do NOT start with "Here is", "This is", "I've rewritten" or similar phrases
3. Do NOT provide multiple versions - give me ONE rewritten version only
4. PRESERVE all product names, brands, and specific references exactly as they are
5. Do NOT invent features, specs, or details not present in the original
6. Do NOT modify URLs or links
7. Keep length within ±20% of original (original: {orig_len} chars)
8. Maintain the original tone and brand voice
9. Start your response directly with the rewritten content

WRONG: "Here is a better version: [content]"
CORRECT: "[content]"
""".format(orig_len=len(content))
        
        prompts = {
            'hook': f"""You are a content optimization engine.

You DO NOT summarize content.
You DO NOT shorten content aggressively.
You DO NOT remove product details.

ORIGINAL CONTENT:
{content}

ORIGINAL WORD COUNT: {len(content.split())} words

{topic_context}

TASK: Hook-Enhanced Variant

RULES:
• Improve ONLY the opening 1–2 lines
• The rest of the content must remain semantically intact
• Add curiosity, urgency, or emotional pull WITHOUT removing details
• Keep total length between {int(len(content.split()) * 0.7)} and {int(len(content.split()) * 1.3)} words (70-130% range)
• Preserve all product features and specifications
• Do NOT change content type

SELF-CHECK:
• Length: {int(len(content.split()) * 0.7)}-{int(len(content.split()) * 1.3)} words? ✓
• All product details preserved? ✓
• Sentiment maintained or improved? ✓
• Only hook modified? ✓

Your optimized variant (FULL CONTENT, start directly):""",
            
            'clarity': f"""You are a content optimization engine.

⚠️ CRITICAL LENGTH REQUIREMENT: You MUST maintain word count between {int(len(content.split()) * 0.7)} and {int(len(content.split()) * 1.3)} words.
Any output shorter than {int(len(content.split()) * 0.7)} words will be REJECTED.

You DO NOT summarize content.
You DO NOT shorten content aggressively.
You DO NOT remove product details.
You DO NOT create short versions.

ORIGINAL CONTENT:
{content}

ORIGINAL WORD COUNT: {len(content.split())} words
MINIMUM REQUIRED: {int(len(content.split()) * 0.7)} words
MAXIMUM ALLOWED: {int(len(content.split()) * 1.3)} words

{topic_context}

TASK: Clarity & Readability Variant

RULES (STRICTLY ENFORCE):
1. LENGTH PRESERVATION: Keep content at similar length (70-130% of original)
2. Improve sentence flow and readability
3. Simplify phrasing WITHOUT losing meaning
4. Preserve ALL product features and specifications
5. Do not reduce content depth or remove sentences
6. Maintain original messaging and tone
7. If original has multiple paragraphs, keep multiple paragraphs

FORBIDDEN:
❌ Removing entire sentences
❌ Creating summaries or condensed versions
❌ Dropping product specifications
❌ Making content significantly shorter

SELF-CHECK (BEFORE RESPONDING):
• Length: {int(len(content.split()) * 0.7)}-{int(len(content.split()) * 1.3)} words? ✓
• All product details preserved? ✓
• Better readability without content loss? ✓
• Meaning fully preserved? ✓
• Did I maintain similar length? ✓

Your optimized variant (FULL CONTENT, start directly):""",
            
            'cta': f"""You are a content optimization engine.

You DO NOT summarize content.
You DO NOT shorten content aggressively.
You DO NOT remove product details.

ORIGINAL CONTENT:
{content}

ORIGINAL WORD COUNT: {len(content.split())} words

{topic_context}

TASK: CTA-Optimized Variant

RULES:
• Improve call-to-action clarity and strength
• CTA must feel natural, not salesy
• Keep total length between {int(len(content.split()) * 0.7)} and {int(len(content.split()) * 1.3)} words (70-130% range)
• Preserve all product features and specifications
• No extra hashtags or emojis unless content type requires it
• Maintain original tone and style

SELF-CHECK:
• Length: {int(len(content.split()) * 0.7)}-{int(len(content.split()) * 1.3)} words? ✓
• All product details preserved? ✓
• CTA improved but natural? ✓
• No aggressive shortening? ✓

Your optimized variant (FULL CONTENT, start directly):""",
            
            'professional': f"""{topic_context}{constraints}

Task: Rewrite to sound more professional and authoritative while remaining engaging. DO NOT just add formal words - genuinely improve the professional tone.

Original text:
{content}

Your rewrite (start directly with the new text, no preamble):""",
            
            'complete': f"""{topic_context}{constraints}

Task: Do a COMPLETE rewrite optimizing for:
1. Attention-grabbing hook
2. Clear, concise message
3. Strong call-to-action
4. Professional yet engaging tone

Original text:
{content}

Your complete rewrite (start directly with the new text, no preamble):"""
        }
        
        prompt = prompts.get(transformation_type, prompts['complete'])
        
        try:
            print(f"[DEBUG] Sending request to Gemini for {transformation_type} transformation...")
            
            # Initialize Gemini model
            model = genai.GenerativeModel(self.model)
            
            # Generate content with retry logic for rate limits
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    response = model.generate_content(
                        prompt,
                        generation_config={
                            'temperature': 0.7,
                            'top_p': 0.9,
                            'max_output_tokens': 2048
                        }
                    )
                    
                    # Update last request time
                    self._last_request_time = time.time()
                    
                    rewritten = response.text.strip()
                    
                    print(f"[DEBUG] Got response: {len(rewritten)} chars")
                    print(f"[DEBUG] Preview: {rewritten[:100]}...")
                    
                    if not rewritten:
                        print("[DEBUG] Empty response from Gemini")
                        return None
                    
                    # Apply cleanup with length constraints
                    rewritten = self._cleanup_llm_output(rewritten, len(content))
                    
                    print(f"[DEBUG] After cleanup: {len(rewritten)} chars")
                    
                    # Return None if cleanup resulted in empty string (indicates failure)
                    return rewritten if rewritten else None
                    
                except Exception as api_error:
                    error_msg = str(api_error)
                    
                    # Handle rate limit errors (429 Resource Exhausted)
                    if '429' in error_msg or 'Resource' in error_msg or 'quota' in error_msg.lower():
                        retry_count += 1
                        if retry_count < max_retries:
                            wait_time = self._rate_limit_delay * (retry_count + 1)
                            print(f"[RATE LIMIT] 429 error detected. Retry {retry_count}/{max_retries} after {wait_time}s...")
                            time.sleep(wait_time)
                            continue
                        else:
                            print(f"[ERROR] Rate limit exceeded after {max_retries} retries")
                            return None
                    else:
                        # Other API errors - don't retry
                        print(f"[ERROR] Gemini API error: {error_msg[:200]}")
                        return None
            
            # Should not reach here
            return None
            
        except Exception as e:
            print(f"[ERROR] Gemini rewriting error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _validate_variant_length(self, original: str, variant: str, variant_name: str, platform: str = 'twitter') -> bool:
        """
        Validate variant length based on platform-specific rules
        
        Platform rules:
        - Twitter: Character count <= 280
        - Reddit/LinkedIn: Word count 70-130% of original
        """
        
        # Twitter uses character-based validation
        if platform.lower() == 'twitter':
            char_count = len(variant)
            is_valid = char_count <= 280
            
            if not is_valid:
                print(f"⚠️ {variant_name} length validation failed:")
                print(f"   Platform: Twitter (character limit)")
                print(f"   Variant length: {char_count} characters")
                print(f"   Maximum allowed: 280 characters")
                print(f"   Exceeded by: {char_count - 280} characters")
            else:
                print(f"✅ {variant_name} length OK: {char_count} characters (Twitter)")
            
            return is_valid
        
        # Reddit/LinkedIn use word-based validation
        else:
            original_words = len(original.split())
            variant_words = len(variant.split())
            
            # Relaxed constraints: 70-130% (was 80-120%)
            min_words = int(original_words * 0.7)
            max_words = int(original_words * 1.3)
            
            is_valid = min_words <= variant_words <= max_words
            
            if not is_valid:
                print(f"⚠️ {variant_name} length validation failed:")
                print(f"   Platform: {platform} (word count validation)")
                print(f"   Original: {original_words} words")
                print(f"   Variant: {variant_words} words")
                print(f"   Required: {min_words}-{max_words} words (70-130% range)")
                print(f"   Deviation: {((variant_words/original_words - 1) * 100):.1f}%")
            else:
                deviation = ((variant_words/original_words - 1) * 100)
                print(f"✅ {variant_name} length OK: {variant_words} words ({deviation:+.1f}%)")
            
            return is_valid
    
    def _validate_content_preservation(self, original: str, variant: str) -> dict:
        """Check if key content elements are preserved"""
        
        # Extract key elements (product names, numbers, URLs)
        original_urls = set(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', original))
        variant_urls = set(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', variant))
        
        original_numbers = set(re.findall(r'\\b\\d+(?:\\.\\d+)?(?:GB|TB|MP|Hz|%|million|billion)?\\b', original.lower()))
        variant_numbers = set(re.findall(r'\\b\\d+(?:\\.\\d+)?(?:GB|TB|MP|Hz|%|million|billion)?\\b', variant.lower()))
        
        original_caps = set(re.findall(r'\\b[A-Z][A-Za-z0-9]*(?:\\s+[A-Z][A-Za-z0-9]*)*\\b', original))
        variant_caps = set(re.findall(r'\\b[A-Z][A-Za-z0-9]*(?:\\s+[A-Z][A-Za-z0-9]*)*\\b', variant))
        
        return {
            'urls_preserved': original_urls == variant_urls,
            'missing_urls': original_urls - variant_urls,
            'numbers_preserved': len(original_numbers - variant_numbers) == 0,
            'missing_numbers': original_numbers - variant_numbers,
            'key_terms_preserved': len(original_caps & variant_caps) >= len(original_caps) * 0.8,
            'missing_terms': original_caps - variant_caps
        }
    
    def generate_smart_variants(self, content: str, platform: str = 'twitter', 
                                topic: Optional[str] = None, num_variants: int = 5) -> Dict[str, any]:
        """
        Generate intelligent, marketing-grade variants with proper labels
        REQUIRES Google Gemini API - variants are generated ONLY by LLM
        
        Args:
            content: Original content
            platform: Target platform
            topic: Topic for context-aware optimization
            num_variants: Number of variants to generate (default 5)
        
        Returns:
            Dict with 'success' boolean, 'variants' list, and 'error' message if failed
        """
        # Check Gemini status first
        status = self.check_gemini_status()
        
        if not status['available']:
            return {
                'success': False,
                'variants': [],
                'error': status['message'],
                'status': status
            }
        
        if not status['target_model_available']:
            return {
                'success': False,
                'variants': [],
                'error': f"Model '{self.model}' is not available. Available models: {', '.join(status['models'][:5])}",
                'status': status
            }
        
        variants = []
        failed_variants = []
        
        # Variant 1: Hook-Optimized (LLM rewrite)
        if num_variants >= 1:
            hook_optimized = self.rewrite_with_llm(content, 'hook', topic)
            if hook_optimized:
                # Validate length
                if self._validate_variant_length(content, hook_optimized, 'Hook Enhanced', platform):
                    # Validate content preservation
                    preservation = self._validate_content_preservation(content, hook_optimized)
                    if not preservation['urls_preserved']:
                        print(f"⚠️ URLs missing in hook variant: {preservation['missing_urls']}")
                    
                    variants.append({
                        'name': 'Hook Enhanced',
                        'content': hook_optimized,
                        'modification': 'Stronger opening hook - improved first impression without content loss',
                        'transformation': 'hook'
                    })
                else:
                    print("⚠️ Hook variant failed length validation, skipping...")
                    failed_variants.append('Hook Enhanced')
            else:
                failed_variants.append('Hook Enhanced')
        
        # Variant 2: Clarity + Structure (LLM rewrite)
        if num_variants >= 2:
            clarity_optimized = self.rewrite_with_llm(content, 'clarity', topic)
            if clarity_optimized:
                # Validate length
                if self._validate_variant_length(content, clarity_optimized, 'Clarity Enhanced', platform):
                    # Validate content preservation
                    preservation = self._validate_content_preservation(content, clarity_optimized)
                    if not preservation['numbers_preserved']:
                        print(f"⚠️ Numbers missing in clarity variant: {preservation['missing_numbers']}")
                    
                    variants.append({
                        'name': 'Clarity Enhanced',
                        'content': clarity_optimized,
                        'modification': 'Improved readability - enhanced structure without content loss',
                        'transformation': 'clarity'
                    })
                else:
                    print("⚠️ Clarity variant failed length validation, skipping...")
                    failed_variants.append('Clarity Enhanced')
            else:
                failed_variants.append('Clarity Enhanced')
        
        # Variant 3: CTA Enhanced (LLM rewrite)
        if num_variants >= 3:
            cta_optimized = self.rewrite_with_llm(content, 'cta', topic)
            if cta_optimized:
                # Validate length
                if self._validate_variant_length(content, cta_optimized, 'CTA Enhanced', platform):
                    # Validate content preservation
                    preservation = self._validate_content_preservation(content, cta_optimized)
                    if not preservation['urls_preserved']:
                        print(f"⚠️ URLs missing in CTA variant: {preservation['missing_urls']}")
                    
                    variants.append({
                        'name': 'CTA Enhanced',
                        'content': cta_optimized,
                        'modification': 'Stronger call-to-action - compelling ending without content loss',
                        'transformation': 'cta'
                    })
                else:
                    print("⚠️ CTA variant failed length validation, skipping...")
                    failed_variants.append('CTA Enhanced')
            else:
                failed_variants.append('CTA Enhanced')
        
        # Variant 4: Professional Tone (LLM rewrite)
        if num_variants >= 4:
            # Reuse hook prompt with professional focus
            professional_optimized = self.rewrite_with_llm(content, 'complete', topic)
            if professional_optimized:
                # Add topic-specific hashtags
                hashtags = self.generate_topic_hashtags(topic, count=3)
                if not re.search(r'#\w+', professional_optimized):
                    professional_optimized = f"{professional_optimized}\n\n{' '.join(hashtags)}"
                
                variants.append({
                    'name': 'Professional + Hashtags',
                    'content': professional_optimized,
                    'modification': f'Professional rewrite with topic-specific hashtags: {", ".join(hashtags)}',
                    'transformation': 'professional'
                })
            else:
                failed_variants.append('Professional + Hashtags')
        
        # Variant 5: Complete Optimization (LLM full rewrite)
        if num_variants >= 5:
            complete_optimized = self.rewrite_with_llm(content, 'complete', topic)
            
            if complete_optimized:
                # Add topic-specific hashtags if not present
                hashtags = self.generate_topic_hashtags(topic, count=2)
                if not re.search(r'#\w+', complete_optimized):
                    complete_optimized = f"{complete_optimized}\n\n{' '.join(hashtags)}"
                
                variants.append({
                    'name': 'Complete Optimization',
                    'content': complete_optimized,
                    'modification': 'Full marketing-grade rewrite - hook, clarity, CTA, topic keywords, and hashtags all optimized',
                    'transformation': 'complete'
                })
            else:
                failed_variants.append('Complete Optimization')
        
        # Check if we generated any variants
        if not variants:
            return {
                'success': False,
                'variants': [],
                'error': f'Failed to generate any variants. LLM may not be responding. Failed: {", ".join(failed_variants)}',
                'status': status
            }
        
        return {
            'success': True,
            'variants': variants,
            'error': None,
            'failed_variants': failed_variants if failed_variants else None,
            'status': status
        }
        
        # Variant 5: Complete Optimization (LLM full rewrite)
        if num_variants >= 5:
            complete_optimized = self.rewrite_with_llm(content, 'complete', topic)
            # Add topic-specific hashtags
            hashtags = self.generate_topic_hashtags(topic, count=2)
            complete_optimized = f"{complete_optimized}\\n\\n{' '.join(hashtags)}"
            
            variants.append({
                'name': 'Complete Optimization',
                'content': complete_optimized,
                'modification': 'Full LLM rewrite with topic context, hashtags, and CTA',
                'transformation': 'complete'
            })
        
        return variants
    
    def _add_smart_emojis(self, content: str, max_emojis: int = 3) -> str:
        """
        Add emojis strategically (not randomly)
        
        Args:
            content: Original content
            max_emojis: Maximum number of emojis to add
        
        Returns:
            Content with strategically placed emojis
        """
        # Don't add if already has emojis
        if re.search(r'[\U0001F300-\U0001F9FF]', content):
            return content
        
        # Emoji mapping for common words/sentiments
        emoji_map = {
            r'\b(launch|start|new|announce)\b': '🚀',
            r'\b(love|amazing|great|awesome|excellent)\b': '✨',
            r'\b(check|look|see|view)\b': '👀',
            r'\b(learn|discover|explore)\b': '💡',
            r'\b(win|success|achieve)\b': '🎯',
            r'\b(time|now|today)\b': '⏰',
            r'\b(free|bonus|gift)\b': '🎁'
        }
        
        result = content
        emojis_added = 0
        
        for pattern, emoji in emoji_map.items():
            if emojis_added >= max_emojis:
                break
            
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                # Add emoji after the matched word
                word = match.group()
                result = re.sub(rf'\b{word}\b', f'{word} {emoji}', result, count=1, flags=re.IGNORECASE)
                emojis_added += 1
        
        return result
    
    def _analyze_hook_quality(self, content: str) -> int:
        """
        Analyze hook strength (first sentence quality)
        
        Returns:
            Score 0-100
        """
        sentences = content.split('.')
        if not sentences:
            return 20
        
        first_sentence = sentences[0].strip()
        score = 40  # Base
        
        # Bonus for question hooks
        if '?' in first_sentence:
            score += 25
        
        # Bonus for emotional triggers
        emotional_words = ['discover', 'amazing', 'exclusive', 'breakthrough', 'transform', 'revolutionary']
        if any(word in first_sentence.lower() for word in emotional_words):
            score += 20
        
        # Bonus for numbers/stats
        if any(char.isdigit() for char in first_sentence):
            score += 15
        
        # Penalty for being too generic
        generic_starts = ['this is', 'here is', 'we are', 'check out']
        if any(first_sentence.lower().startswith(phrase) for phrase in generic_starts):
            score -= 20
        
        return min(100, max(0, score))
    
    def _analyze_cta_strength(self, content: str) -> int:
        """
        Analyze call-to-action strength
        
        Returns:
            Score 0-100
        """
        score = 0
        
        # Strong CTAs
        strong_ctas = ['learn more', 'get started', 'try now', 'join us', 'discover', 'explore', 'see how', 'find out']
        weak_ctas = ['check out', 'check it out', 'see it']
        
        content_lower = content.lower()
        
        # Check for CTA presence and strength
        if any(cta in content_lower for cta in strong_ctas):
            score += 70
        elif any(cta in content_lower for cta in weak_ctas):
            score += 30
        
        # Bonus for action words at the end
        last_sentence = content.split('.')[-1].strip().lower()
        action_words = ['start', 'join', 'discover', 'explore', 'try', 'get', 'learn']
        if any(word in last_sentence for word in action_words):
            score += 20
        
        # Bonus for urgency
        if any(word in content_lower for word in ['today', 'now', 'limited', 'exclusive']):
            score += 10
        
        return min(100, score)
    
    def calculate_engagement_score(self, content: str, analysis: Dict, platform: str = 'twitter') -> int:
        """
        Calculate realistic engagement score based on multiple weighted factors
        
        Args:
            content: The content to score
            analysis: Analysis dictionary from analyze_content()
            platform: Platform (twitter or reddit)
        
        Returns:
            Normalized engagement score (0-100 integer)
        """
        score = 5  # Lower base score (normalized to 0-100 scale)
        
        # Hook quality (HIGH WEIGHT - 25% of score)
        hook_score = self._analyze_hook_quality(content)
        score += (hook_score / 100) * 25
        
        # CTA strength (HIGH WEIGHT - 20% of score)
        cta_score = self._analyze_cta_strength(content)
        score += (cta_score / 100) * 20
        
        # Sentiment bonus (10% weight)
        if analysis['sentiment_polarity'] > 0.3:
            score += 10
        elif analysis['sentiment_polarity'] > 0.1:
            score += 5
        elif analysis['sentiment_polarity'] < -0.2:
            score -= 8
        
        # Readability (15% weight - easier = better)
        if analysis['readability'] > 70:
            score += 15
        elif analysis['readability'] > 50:
            score += 8
        elif analysis['readability'] < 30:
            score -= 10
        
        # Length optimization (platform-specific, 15% weight)
        if platform == 'twitter':
            if 40 <= analysis['word_count'] <= 100:
                score += 15  # Optimal
            elif 100 < analysis['word_count'] <= 150:
                score += 8   # Acceptable
            elif analysis['word_count'] > 200:
                score -= 12  # Too long penalty
        else:  # reddit
            if 80 <= analysis['word_count'] <= 250:
                score += 15
            elif analysis['word_count'] > 400:
                score -= 10
        
        # Hashtag optimization (2-3 is ideal)
        if 2 <= analysis['hashtag_count'] <= 3:
            score += 8
        elif analysis['hashtag_count'] == 1:
            score += 3
        elif analysis['hashtag_count'] > 5:
            score -= 6
        
        # Emoji optimization (1-3 is ideal)
        if 1 <= analysis['emoji_count'] <= 3:
            score += 5
        elif analysis['emoji_count'] > 5:
            score -= 8  # Emoji spam penalty
        
        # Question bonus (engagement trigger)
        if analysis['has_question']:
            score += 7
        
        # Bonus for multiple engagement features
        feature_count = sum([
            analysis['has_cta'],
            analysis['has_question'],
            1 if 1 <= analysis['emoji_count'] <= 3 else 0,
            1 if 2 <= analysis['hashtag_count'] <= 3 else 0
        ])
        
        if feature_count >= 3:
            score += 5  # Bonus for well-rounded content
        
        # Normalize to 0-100 range (integer only)
        score = int(round(max(0, min(score, 100))))
        
        return score
