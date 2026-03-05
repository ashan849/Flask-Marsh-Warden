import requests
import json

BASE_URL = "http://localhost:5000"

def test_endpoint(name, method, url, data=None):
    """Test an endpoint and show detailed response"""
    print("=" * 70)
    print(f"Testing {name}")
    print("=" * 70)
    print(f"URL: {url}")
    print(f"Method: {method}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type')}")
        print(f"Response Length: {len(response.text)} chars")
        print()
        
        # Try to parse as JSON
        try:
            json_data = response.json()
            print("✅ JSON Response:")
            print(json.dumps(json_data, indent=2))
            return True
        except:
            print("❌ NOT JSON! Raw response:")
            print(response.text[:1000])  # First 1000 chars
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR - Server not running on port 5000!")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    print("\n🔍 Flask Server Diagnostic Test")
    print("=" * 70)
    
    # Test 1: Health check
    test_endpoint(
        "Health Check",
        "GET",
        f"{BASE_URL}/api/health"
    )
    
    # Test 2: Sources
    test_endpoint(
        "Sources",
        "GET",
        f"{BASE_URL}/api/sources"
    )
    
    # Test 3: Chat
    test_endpoint(
        "Chat",
        "POST",
        f"{BASE_URL}/api/chat",
        data={"message": "What are the main waste streams?", "session_id": "test"}
    )
    
    print("\n" + "=" * 70)
    print("DEBUGGING TIPS:")
    print("=" * 70)
    print("1. Check if Flask is running: netstat -an | findstr :5000")
    print("2. Check Flask console for error messages")
    print("3. Verify index file exists: dir pdf_index_enhanced1.pkl")
    print("4. Test RAG import: python -c \"from rag_pipeline import RAGPipeline\"")
    print("5. Check .env file has HF_TOKEN or API keys")
    print("=" * 70)

if __name__ == "__main__":
    main()