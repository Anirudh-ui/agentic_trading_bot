import json
import google.generativeai as genai

import os
from dotenv import load_dotenv
load_dotenv()
NER_API_KEY = os.getenv("NER_API_KEY")
genai.configure(api_key=NER_API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")


def extract_entities_with_gemini(text: str) -> dict:
    """
    Uses Gemini Flash to extract structured NER.
    Returns strict JSON only.
    """
    print("🔍 Extracting entities with Gemini NER...")
    prompt = f"""
You are an NER extraction engine. 
Extract ONLY the following entities from the user text.
Return STRICT JSON. No explanation. No extra text.

Fields:
- person_name
- location
- company
- ticker
- other_keywords

If an entity is missing, return null for that field.

TEXT:
{text}

Return JSON only:
{{
  "person_name": ...,
  "location": ...,
  "company": ...,
  "ticker": ...,
  "other_keywords": ...
}}
"""

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()

        # Ensure valid JSON
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        print("✅ Gemini NER Success:", cleaned)
        return json.loads(cleaned)

    except Exception as e:
        print("❌ Gemini NER Error:", e)
        return {
            "person_name": None,
            "location": None,
            "company": None,
            "ticker": None,
            "other_keywords": None
        }
