import requests
import json

url = "http://localhost:5000/api/chat"

# Test different questions
questions = [
    "What is 2+2?",
    "Tell me about Azure",
    "How does capacity planning work?"
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"Question: {q}")
    print('='*60)
    
    response = requests.post(url, json={
        "message": q,
        "system_instruction": "You are a helpful assistant."
    }, timeout=15)
    
    print(f"Status: {response.status_code}")
    reply = response.json().get('reply', 'No reply')
    print(f"Reply: {reply[:200]}")
