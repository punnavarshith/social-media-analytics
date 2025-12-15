import sys
sys.stdout.reconfigure(encoding='utf-8')

from engagement_predictor import EngagementPredictor

print("Testing Gemini JSON parsing...")

ep = EngagementPredictor()
content = "Join us for an exciting webinar on AI! Register now #AI"

try:
    result = ep.predict_engagement(content, "twitter")
    
    print("\nRESULT:")
    print(f"Score: {result['predicted_engagement']}")
    print(f"Confidence: {result['confidence']}")  
    print(f"Reasoning: {result['reasoning'][:100]}")
    
    if "Heuristic" in result['reasoning']:
        print("\nSTATUS: Still using fallback")
    else:
        print("\nSTATUS: SUCCESS - Using Gemini!")
except Exception as e:
    print(f"\nERROR: {e}")
