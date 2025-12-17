import google.generativeai as genai
import os
from dotenv import load_dotenv

# 1. Load the .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: Could not find 'GEMINI_API_KEY' in your .env file.")
else:
    print("API Key found. Connecting to Google...\n")
    genai.configure(api_key=api_key)

    try:
        print("--- AVAILABLE MODELS FOR YOUR KEY ---")
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                # We only print the ID (m.name) to avoid errors
                print(f"ID: {m.name}")

    except Exception as e:
        print(f"\nError: {e}")
