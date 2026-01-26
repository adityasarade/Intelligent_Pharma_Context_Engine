import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found.")
    exit(1)

genai.configure(api_key=api_key)

print("Listing models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)
except Exception as e:
    print(f"Error listing models: {e}")

print("\nTesting generation with 'gemini-1.5-flash'...")
try:
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Hello, can you extract name 'Advil' from this text: 'Buy Advil'?")
    print("Response text:", response.text)
except Exception as e:
    print(f"Error with gemini-1.5-flash: {e}")

print("\nTesting generation with 'gemini-pro'...")
try:
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Hello, can you extract name 'Advil' from this text: 'Buy Advil'?")
    print("Response text:", response.text)
except Exception as e:
    print(f"Error with gemini-pro: {e}")
