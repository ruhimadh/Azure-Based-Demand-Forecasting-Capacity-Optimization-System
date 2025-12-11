import google.generativeai as genai
import os

# Load API key from.env
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("No API key found in .env")
    exit(1)

print(f"Using API key: {api_key[:10]}...")

try:
    genai.configure(api_key=api_key)
    
    print("\nAvailable models:")
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"  - {model.name}")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
