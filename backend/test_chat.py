import requests
import json

url = "http://localhost:5000/api/chat"

# Test with a real Azure/forecasting context question
data = {
    "message": "Explain CPU usage forecasting",
    "system_instruction": "You are an Azure capacity planning assistant helping users understand their infrastructure metrics."
}

try:
    print("Testing with context-relevant question...")
    response = requests.post(url, json=data, timeout=15)
    
    # Write full response to file
    with open('chat_response.txt', 'w', encoding='utf-8') as f:
        f.write(f"Status Code: {response.status_code}\n\n")
        f.write("Response JSON:\n")
        response_json = response.json()
        f.write(json.dumps(response_json, indent=2))
        f.write("\n\n")
        if 'reply' in response_json:
            f.write(f"Reply:\n{response_json['reply']}\n")
    
    print(f"Status: {response.status_code}")
    
    # Also print to console
    if response.status_code == 200:
        reply = response.json().get('reply', 'No reply')
        print(f"\n✅ SUCCESS!\n{reply[:300]}")
    else:
        print(f"❌ Error: {response.json().get('reply', 'Unknown error')}")
    
except Exception as e:
    print(f"❌ Connection Error: {e}")
