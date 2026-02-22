"""
CivicLens - Complaint Generator
Uses GPT-4o Vision to analyze the actual image and generate
a rich, specific complaint description.
"""

import os

from openai import OpenAI
import base64
import json

# ─── PUT YOUR OPENAI API KEY HERE ─────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ──────────────────────────────────────────────────────────────────────────────

client = OpenAI(api_key=OPENAI_API_KEY)

PROMPT = """You are a civic complaint assistant. Look at this image carefully and generate a detailed, professional complaint report.

Detected issue type: {class_name}
Severity: {severity}
Department: {department}

Based on what you actually SEE in the image, return a JSON object:
{{
  "title": "specific short title max 10 words",
  "description": "2-3 sentences describing exactly what you see — size, location, condition, immediate risk to public",
  "action_required": "specific action the department should take",
  "priority": "{severity}",
  "department": "{department}"
}}

Be specific — mention visible details like size, depth, extent of damage, location clues.
Return ONLY the JSON. No extra text.
"""

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_complaint(detection: dict, image_path: str = None) -> dict:
    class_name = detection["class"].replace("_", " ")
    severity   = detection["severity"]
    department = detection["department"]

    prompt = PROMPT.format(
        class_name=class_name,
        severity=severity,
        department=department,
    )

    try:
        if image_path:
            # GPT-4o Vision — analyze actual image
            b64 = encode_image(image_path)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        { "type": "text", "text": prompt },
                        { "type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{b64}" } }
                    ]
                }],
                max_tokens=400,
                temperature=0.3,
            )
        else:
            # Fallback to text only if no image
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
            )

        text = response.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)

    except Exception as e:
        print(f"OpenAI error: {e}")
        return {
            "title": f"{class_name.title()} Detected",
            "description": f"A {class_name} has been detected and requires immediate attention from {department}.",
            "action_required": "Please dispatch a team to inspect and repair the issue.",
            "priority": severity,
            "department": department,
        }