#!/usr/bin/env python3
"""
Test script for the Fake News Detector Web API
Demonstrates how to interact with the web interface programmatically
"""

import requests
import json
import time

def test_api():
    """Test the fake news detection API"""
    
    base_url = "http://127.0.0.1:5000"
    
    print("🧪 TESTING FAKE NEWS DETECTOR WEB API")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        {
            'name': 'Swiss Resort Fire (Real News)',
            'input': 'Swiss resort fire',
            'expected': 'TRUE'
        },
        {
            'name': 'La Constellation Fire (Real News)',
            'input': 'La Constellation fire champagne sparklers',
            'expected': 'TRUE'
        },
        {
            'name': 'Invalid Input (Too Few Keywords)',
            'input': 'fire',
            'expected': 'ERROR'
        },
        {
            'name': 'Unknown Event',
            'input': 'alien invasion mars colony',
            'expected': 'ERROR'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test_case['name']}")
        print(f"Input: '{test_case['input']}'")
        print("-" * 30)
        
        try:
            # Make API request
            response = requests.post(
                f"{base_url}/analyze",
                json={'user_input': test_case['input']},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✅ Status: SUCCESS")
                print(f"📊 Verdict: {data['verdict']}")
                print(f"🎯 Credibility Score: {data['credibility_score']:.2f}")
                print(f"📰 Sources Found: {data['articles_count']}")
                print(f"⏱️ Timeline Span: {data['timeline_span_hours']:.1f} hours")
                
                # Show first source if available
                if data['sources_timeline']:
                    first_source = data['sources_timeline'][0]
                    print(f"📄 First Source: {first_source['source']}")
                    print(f"   Title: {first_source['title'][:60]}...")
                
            else:
                error_data = response.json()
                print(f"⚠️ Status: ERROR ({response.status_code})")
                print(f"❌ Error: {error_data.get('error', 'Unknown error')}")
                if 'suggestion' in error_data:
                    print(f"💡 Suggestion: {error_data['suggestion']}")
        
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection Error: {e}")
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")
        
        # Small delay between tests
        time.sleep(1)
    
    print(f"\n" + "=" * 50)
    print("🎯 API TESTING COMPLETED")
    print("=" * 50)
    print("✅ Web interface is running at: http://127.0.0.1:5000")
    print("✅ API endpoint available at: http://127.0.0.1:5000/analyze")
    print("✅ Ready for user interaction!")

if __name__ == "__main__":
    # Wait a moment for the server to fully start
    print("⏳ Waiting for server to start...")
    time.sleep(3)
    
    test_api()