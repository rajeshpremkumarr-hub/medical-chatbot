import requests
import json
import os

def test_openrouter(api_key, model="openai/gpt-4o-mini"):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Medical Assistant Test"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}]
    }
    
    print(f"Testing URL: {url}")
    print(f"Headers (masked key): {headers['Authorization'][:15]}...")
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")

if __name__ == "__main__":
    # I'll just use a placeholder to see if I get a 401 (Missing Header) 
    # instead of a 401 (Invalid Key)
    test_openrouter("sk-or-v1-fake-key")
