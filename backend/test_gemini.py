import requests
import json

url = "http://localhost:5000/api/chat"

# Test with real Azure context
data = {
    "message": "What is 2+2?",
    "system_instruction": "You are an Azure capacity planning assistant."
}

print("Testing simple math question to verify Gemini is responding...")
response = requests.post(url, json=data, timeout=30)

print(f"\nStatus: {response.status_code}")
reply = response.json().get('reply', 'No reply')
print(f"Reply: {reply}")

# Write to file
with open('chat_response.txt', 'w', encoding='utf-8') as f:
    f.write(f"Status Code: {response.status_code}\n\n")
    f.write("Response JSON:\n")
    f.write(json.dumps(response.json(), indent=2))
