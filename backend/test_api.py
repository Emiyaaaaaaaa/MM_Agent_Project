import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_backend():
    print(f"Testing Backend at {BASE_URL}...")
    
    # 1. Heartbeat
    try:
        r = requests.get(f"{BASE_URL}/")
        print(f"Heartbeat: {r.status_code} - {r.json()}")
    except Exception as e:
        print(f"Heartbeat Failed: {e}")
        return

    # 2. Search
    print("\nTesting Search API...")
    payload = {
        "query": "What is the four-quadrant model for cybersecurity policies?",
        "top_k": 1
    }
    try:
        r = requests.post(f"{BASE_URL}/api/v1/search", json=payload)
        print(f"Search Status: {r.status_code}")
        if r.status_code == 200:
            results = r.json()
            if results:
                res = results[0]
                print(f"Top Result ID: {res['id']}")
                print(f"Doc Type: {res.get('doc_type', 'N/A')}")
                print(f"Distance: {res['distance']:.4f}")
                print(f"Images Found: {len(res['image_urls'])}")
                for url in res['image_urls']:
                    print(f"  URL: {url}")
            else:
                print("No results found.")
    except Exception as e:
        print(f"Search Failed: {e}")

if __name__ == "__main__":
    test_backend()
