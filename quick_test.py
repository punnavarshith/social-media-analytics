"""Quick test of JSON parsing fix"""
from engagement_predictor import EngagementPredictor

print("Quick JSON parsing test...")
print("=" * 60)

ep = EngagementPredictor()

# Single test
content = "Join us for an exciting webinar on AI! Register now #AI"
result = ep.predict_engagement(content, "twitter")

print(f"\nResult:")
print(f"Score: {result['predicted_engagement']}")
print(f"Confidence: {result['confidence']}")
print(f"Reasoning: {result['reasoning']}")

# Check success
if "Heuristic" in result['reasoning']:
    print("\nSTILL USING FALLBACK")
else:
    print("\nSUCCESS - Using Gemini AI!")
