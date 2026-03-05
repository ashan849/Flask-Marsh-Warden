import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_query(question, expected_markers=None):
    print(f"\nTesting: {question}")
    try:
        response = requests.post(
            f"{BASE_URL}/api/chat",
            json={"message": question, "session_id": f"test-{int(time.time())}"},
            timeout=120
        )
        if response.status_code == 200:
            content = response.json().get("response", "")
            print("Response received.")
            
            # Check for formatting markers
            checks = []
            if expected_markers:
                for marker in expected_markers:
                    if marker in content:
                        checks.append(f"✅ Found {marker}")
                    else:
                        checks.append(f"❌ Missing {marker}")
            
            print("\n".join(checks))
            print("-" * 40)
            return True
        else:
            print(f"Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Starting Accuracy and Formatting Verification...")
    
    # Test 1: Requires Table
    test_query(
        "List and compare the penalties for different wetland violations in a table.",
        expected_markers=["|", "##", "Sources Used", "["]
    )
    
    # Test 2: Informative Bullets
    test_query(
        "Explain the protection rules for the Muturajawela wetland in detail with bullet points.",
        expected_markers=["-", "##", "Sources Used", "["]
    )
