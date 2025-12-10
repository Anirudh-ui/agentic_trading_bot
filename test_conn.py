import os
from dotenv import load_dotenv
from PIL import Image
import io

from vertexai import init
from vertexai.generative_models import GenerativeModel, Part

load_dotenv()

print("Using credentials:", os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
print("\nTesting NEW Gemini API...\n")

# 1. INIT Vertex AI
init(
    project=os.getenv("DOC_AI_PROJECT_ID"),
    location="us-central1"
)

# 2. Load Gemini Flash
model = GenerativeModel("gemini-2.5-flash-lite")
print("Model loaded successfully!\n")

# -------------------------
# TEXT TEST
# -------------------------
try:
    res = model.generate_content("Say hello if you are working.")
    print("✅ TEXT TEST SUCCESS:\n", res.text, "\n")
except Exception as e:
    print("❌ TEXT TEST FAILED!\nError:\n", e)

# -------------------------
# IMAGE TEST (NEW API)
# -------------------------
try:
    # Create simple image in memory
    img = Image.new("RGB", (200, 200), color=(255, 0, 0))

    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    # Wrap in Part
    img_part = Part.from_data(
        mime_type="image/png",
        data=img_bytes
    )

    response = model.generate_content([
        "What color is this image?",
        img_part
    ])

    print("✅ IMAGE TEST SUCCESS:\n", response.text)

except Exception as e:
    print("❌ IMAGE TEST FAILED!\nError:\n", e)
