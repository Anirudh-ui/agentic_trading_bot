"""
Gemini Multimodal Document Processor (FINAL VERSION)
- Text extraction (PDF/DOCX)
- OCR text extraction (Gemini)
- Chart + Diagram extraction (Gemini)
- Page-level multimodal analysis
- Store vectors in Pinecone for RAG
"""

import os
import json
import re
from uuid import uuid4
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List

from PIL import Image
import pymupdf          # For PDF text extraction
from pdf2image import convert_from_path
import docx

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from custom_logging.my_logger import logger


# ============================================================
#   GEMINI VISION ANALYZER (Charts, OCR, Diagrams)
# ============================================================

class GeminiVisionAnalyzer:

    def __init__(self):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel("gemini-1.5-flash")

        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }

    def analyze_page(self, image_bytes: bytes, page_num: int) -> Dict[str, Any]:
        """
        Vision analysis:
        - Charts
        - Diagrams
        - OCR text inside image
        """
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        prompt = f"""
You are an expert vision analyzer for documents.

TASKS:
1. Detect CHARTS (bar, line, pie, scatter, any data visual)
2. Detect DIAGRAMS / ILLUSTRATIONS
3. Extract OCR TEXT that appears inside the image

RETURN ONLY JSON in this format:

{{
  "charts": [
    {{
      "chart_id": "C1",
      "type": "bar/line/pie/other",
      "description": "What the chart shows",
      "key_values": "Important numeric values or trends",
      "page_num": {page_num}
    }}
  ],
  "images": [
    {{
      "image_id": "I1",
      "description": "Description of diagram/image",
      "text_inside": "OCR extracted text",
      "page_num": {page_num}
    }}
  ]
}}

Rules:
- If no charts, return empty list.
- If no images, return empty list.
- DO NOT include markdown. DO NOT include commentary.
"""

        try:
            response = self.model.generate_content(
                [prompt, image],
                safety_settings=self.safety_settings,
            )
            raw = response.text.strip()
            raw = re.sub(r"```.*?```", "", raw, flags=re.DOTALL).strip()

            parsed = json.loads(raw)

        except Exception as e:
            logger.warning(f"[VISION] JSON failed, fallback used: {e}")
            parsed = {"charts": [], "images": []}

        return parsed


# ============================================================
#   MULTIMODAL DOCUMENT PROCESSOR
# ============================================================

class GeminiMultimodalProcessor:

    def __init__(self):
        logger.info("[INGEST] Initializing Gemini ingestion pipeline...")

        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.vision = GeminiVisionAnalyzer()

        # embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )

        # pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index("trading-bot")

    # --------------------------------------------------------
    # PROCESS DOCUMENT ENTRYPOINT
    # --------------------------------------------------------
    def process_document(self, file_path: str) -> Dict[str, Any]:

        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            return self._process_pdf(file_path)
        elif ext == ".docx":
            return self._process_docx(file_path)
        else:
            raise Exception(f"Unsupported file type: {ext}")

    # --------------------------------------------------------
    # PDF PROCESSING
    # --------------------------------------------------------
    def _process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        logger.info(f"[PDF] Processing {pdf_path}")

        pdf = pymupdf.open(pdf_path)
        page_count = len(pdf)

        # Render pages as images (for vision analysis)
        images = convert_from_path(pdf_path, dpi=150)

        text_chunks = []
        charts = []
        images_info = []

        for page_num, (page, img) in enumerate(zip(pdf, images), start=1):

            logger.info(f"[PDF] Page {page_num}/{page_count}")

            # -------- TEXT EXTRACTION --------
            text = page.get_text("text")
            if text.strip():
                text_chunks.append({
                    "page_num": page_num,
                    "content": text,
                    "type": "text"
                })

            # -------- VISION ANALYSIS --------
            img_bytes = BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes = img_bytes.getvalue()

            vision_result = self.vision.analyze_page(img_bytes, page_num)

            charts.extend(vision_result.get("charts", []))
            images_info.extend(vision_result.get("images", []))

        pdf.close()

        logger.info(f"[PDF DONE] Text chunks={len(text_chunks)}, Charts={len(charts)}, Images={len(images_info)}")

        return {
            "text_chunks": text_chunks,
            "tables": [],     # (We deliberately skip Gemini tables)
            "charts": charts,
            "images": images_info,
            "page_count": page_count,
        }

    # --------------------------------------------------------
    # DOCX PROCESSING
    # --------------------------------------------------------
    def _process_docx(self, docx_path: str) -> Dict[str, Any]:
        logger.info(f"[DOCX] Processing {docx_path}")

        doc = docx.Document(docx_path)

        full_text = []
        for p in doc.paragraphs:
            if p.text.strip():
                full_text.append(p.text)

        text_chunks = [{
            "page_num": 1,
            "content": "\n".join(full_text),
            "type": "text"
        }]

        logger.info(f"[DOCX DONE] Extracted {len(full_text)} lines")

        return {
            "text_chunks": text_chunks,
            "tables": [],
            "charts": [],
            "images": [],
            "page_count": 1,
        }

    # --------------------------------------------------------
    # PINECONE STORAGE
    # --------------------------------------------------------
    def store_in_pinecone(self, processed: Dict[str, Any], doc_id: str, filename: str):
        logger.info(f"[PINECONE] Storing vectors for doc {doc_id}")

        vector_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            namespace="documents"
        )

        docs = []

        # ----- TEXT -----
        for chunk in processed["text_chunks"]:
            docs.append(Document(
                page_content=chunk["content"],
                metadata={
                    "doc_id": doc_id,
                    "filename": filename,
                    "type": "text",
                    "page_num": chunk["page_num"]
                }
            ))

        # ----- CHARTS -----
        for chart in processed["charts"]:
            desc = f"{chart.get('description','')}\nKey Values: {chart.get('key_values','')}"
            docs.append(Document(
                page_content=desc,
                metadata={
                    "doc_id": doc_id,
                    "filename": filename,
                    "type": "chart",
                    "page_num": chart["page_num"],
                    "chart_id": chart["chart_id"],
                    "chart_name": chart.get("type", "chart")
                }
            ))

        # ----- IMAGES / DIAGRAMS -----
        for img in processed["images"]:
            desc = f"{img.get('description','')}\nOCR: {img.get('text_inside','')}"
            docs.append(Document(
                page_content=desc,
                metadata={
                    "doc_id": doc_id,
                    "filename": filename,
                    "type": "image",
                    "page_num": img["page_num"],
                    "image_id": img["image_id"]
                }
            ))

        # Generate IDs
        ids = [str(uuid4()) for _ in docs]

        # Upload
        vector_store.add_documents(documents=docs, ids=ids)

        logger.info(f"[PINECONE DONE] Stored {len(docs)} vectors.")


# EXPORT
__all__ = ["GeminiMultimodalProcessor"]
