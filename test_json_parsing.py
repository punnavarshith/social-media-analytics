"""Test improved JSON parsing in EngagementPredictor"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from engagement_predictor import EngagementPredictor

def test_single_prediction():
    """Test a single prediction to verify JSON parsing works"""
    print("Testing improved JSON parsing...")
    print("=" * 60)
    
    ep = EngagementPredictor()
    
    # Test content
    content = "🚀 Excited to share our new AI platform! #AI #Tech"
    platform = "twitter"
    
    print(f"Content: {content}")
    print(f"Platform: {platform}")
    print("=" * 60)
    
    # Make prediction
    result = ep.predict_engagement(content, platform)
    
    print("\nRESULTS:")
    print(f"Score: {result['predicted_engagement']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Sentiment: {result['sentiment_label']}")
    print(f"Quality: {result['quality_score']}/5")
    print(f"Improvement: {result['improvement_potential']}%")
    print(f"Reasoning: {result['reasoning']}")
    print("=" * 60)
    
    # Check if it used fallback
    if "Heuristic" in result['reasoning']:
        print("FAILED: Still using fallback method")
        return False
    else:
        print("SUCCESS: Using Gemini AI predictions")
        return True

def test_multiple_predictions():
    """Test multiple predictions to verify consistency"""
    print("\nTesting multiple predictions...")
    print("=" * 60)
    
    ep = EngagementPredictor()
    
    test_cases = [
        ("Check out this amazing product! 🔥 #innovation", "twitter"),
        ("What are your thoughts on remote work policies?", "reddit"),
        ("Join us for a live webinar tomorrow! Register now 👉", "twitter"),
    ]
    
    success_count = 0
    
    for i, (content, platform) in enumerate(test_cases, 1):
        print(f"\nTest {i}/{len(test_cases)}: {content[:50]}...")
        result = ep.predict_engagement(content, platform)
        
        if "Heuristic" not in result['reasoning']:
            print(f"OK Score: {result['predicted_engagement']} (Gemini)")
            success_count += 1
        else:
            print(f"FAIL Score: {result['predicted_engagement']} (Fallback)")
    
    print("\n" + "=" * 60)
    print(f"Success Rate: {success_count}/{len(test_cases)} ({success_count/len(test_cases)*100:.0f}%)")
    print("=" * 60)
    
    return success_count == len(test_cases)

if __name__ == "__main__":
    # Run tests
    single_success = test_single_prediction()
    multiple_success = test_multiple_predictions()
    
    print("\n" + "=" * 60)
    if single_success and multiple_success:
        print("ALL TESTS PASSED - JSON parsing fixed!")
    else:
        print("SOME TESTS FAILED - needs more work")
    print("=" * 60)
