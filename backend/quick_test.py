import requests

# Quick test to see what model backend logs show
response = requests.post(
    "http://localhost:5000/api/chat",
    json={"message": "hi", "system_instruction": ""},
    timeout=10
)
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
