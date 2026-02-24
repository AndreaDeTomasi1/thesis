import os
import requests
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# ===== CONFIG =====
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "gpt-4o-mini"
INPUT_FILE = "main.typ"
OUTPUT_FILE = "main_en.typ"

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not set")

# ===== READ FILE =====
text = Path(INPUT_FILE).read_text(encoding="utf-8")

# ===== PROMPT =====
system_prompt = """You are a professional academic translator.
Translate the following Typst document from Italian to English.

IMPORTANT:
- Preserve ALL Typst syntax.
- Do NOT modify formatting.
- Do NOT translate code blocks.
- Do NOT add explanations.
- Output ONLY the translated Typst file.
"""

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ],
    "temperature": 0.7
}

response = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    },
    json=payload
)

response.raise_for_status()
translated = response.json()["choices"][0]["message"]["content"]

# ===== WRITE OUTPUT =====
Path(OUTPUT_FILE).write_text(translated, encoding="utf-8")

print("Translation completed: main_en.typ created.")