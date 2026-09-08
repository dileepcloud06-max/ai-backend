import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Hello World! Explain consumer feedback AI review intelligence in 2 short sentences."
    )
    print("Gemini Response:")
    print(response.text)
else:
    print("GEMINI_API_KEY missing in .env")