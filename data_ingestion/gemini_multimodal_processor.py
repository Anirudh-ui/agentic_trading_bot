import os
import json
import re
from uuid import uuid4
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, List

from PIL import Image
import pymupdf
import docx
from pdf2image import convert_from_path

# -----------------------------
# Vertex AI (NEW GEMINI API)
# -----------------------------
from vertexai import init
from vertexai.generative_models import GenerativeModel, Part
from vertexai.language_models import TextEmbeddingModel

# -----------------------------
# Pinecone + LangChain wrapper
# -----------------------------
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Logger
from custom_logging.my_logger import logger


# ============================================================
#               INIT VERTEX AI
# ============================================================

init(
    project=os.getenv("DOC_AI_PROJECT_ID"),
    location="us-central1"
)


# ============================================================
#     LangChain-Compatible Wrapper for Vertex Embeddings
# ============================================================

class VertexAIEmbeddings(Embeddings):
    """Adapter to make Vertex `text-embedding-004` compatible with LangChain."""

    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        results = self.model.get_embeddings(texts)
        return [emb.values for emb in results]

    def embed_query(self, text: str) -> List[float]:
        result = self.model.get_embeddings([text])[0]
        return result.values


# ============================================================
#       GEMINI VISION ANALYZER (OCR + Charts + Tables)
# ============================================================

class GeminiVisionAnalyzer:
    def __init__(self):
        # Fast + cheap vision model
        self.model = GenerativeModel("gemini-2.5-flash-lite")

    def _safe_json(self, text: str):
        """Clean invalid JSON returned by LLM."""
        try:
            cleaned = text.strip()
            cleaned = cleaned.replace("```json", "").replace("```", "")
            return json.loads(cleaned)
        except Exception:
            return {"charts": [], "images": [], "tables": []}

    # -------------------------
    # OCR + Charts + Images
    # -------------------------
    def analyze_page(self, image_bytes: bytes, page_num: int) -> Dict[str, Any]:
        try:
            img_part = Part.from_data(
                mime_type="image/png",
                data=image_bytes
            )

            prompt = """
Return ONLY the following JSON:
{
 "charts": [],
 "images": []
}
No explanations.
No markdown.
"""

            response = self.model.generate_content([prompt, img_part])
            return self._safe_json(response.text)

        except Exception as e:
            logger.warning(f"[VISION] Failed page analysis: {e}")
            return {"charts": [], "images": []}

    # -------------------------
    # Table Extraction
    # -------------------------
    def extract_tables(self, image_bytes: bytes, page_num: int) -> List[Dict]:
        try:
            img_part = Part.from_data(
                mime_type="image/png",
                data=image_bytes
            )

            prompt = f"""
Extract ALL TABLES in this image.
Return JSON ONLY:
{{
 "tables": [
   {{
     "table_id": "T{page_num}_1",
     "page_num": {page_num},
     "description": "Short description",
     "rows": [
        ["Header1", "Header2"]
     ]
   }}
 ]
}}
"""

            response = self.model.generate_content([prompt, img_part])
            parsed = self._safe_json(response.text)
            return parsed.get("tables", [])

        except Exception:
            return []


# ============================================================
#           MAIN MULTIMODAL PROCESSOR CLASS
# ============================================================

class GeminiMultimodalProcessor:

    def __init__(self):
        logger.info("[INGEST] Initializing Gemini ingestion pipeline...")

        # Gemini Vision
        self.vision = GeminiVisionAnalyzer()

        # Embedding model (Vertex AI)
        self.embedder = TextEmbeddingModel.from_pretrained("text-embedding-004")

        # Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index("trading-bot")

    # --------------------------------------------------------
    # PROCESS DOCUMENT ENTRY POINT
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
    # PDF Processing
    # --------------------------------------------------------
    def _process_pdf(self, pdf_path: str):
        logger.info(f"[PDF] Processing {pdf_path}")

        pdf = pymupdf.open(pdf_path)
        page_count = len(pdf)

        images = convert_from_path(pdf_path, dpi=150)

        text_chunks = []
        all_tables = []
        charts = []
        images_info = []

        for page_num, (page, img) in enumerate(zip(pdf, images), start=1):

            logger.info(f"[PDF] Page {page_num}/{page_count}")

            # Image bytes
            buf = BytesIO()
            img.save(buf, format="PNG")
            page_bytes = buf.getvalue()

            # --------------------- OCR ----------------------
            vision_output = self.vision.analyze_page(page_bytes, page_num)

            extracted_text = []
            for item in vision_output.get("images", []):
                if "text_inside" in item:
                    extracted_text.append(item["text_inside"])

            text = "\n".join(extracted_text).strip()

            # Fallback to embedded PDF text
            if not text:
                text = page.get_text("text") or ""

            if text:
                text_chunks.append({
                    "page_num": page_num,
                    "content": text,
                    "type": "text"
                })

            # --------------------- Charts + Images ----------------------
            charts.extend(vision_output.get("charts", []))
            images_info.extend(vision_output.get("images", []))

            # --------------------- Table Extraction ----------------------
            tables = self.vision.extract_tables(page_bytes, page_num)
            all_tables.extend(tables)

        pdf.close()

        logger.info(
            f"[PDF DONE] Text={len(text_chunks)}, "
            f"Tables={len(all_tables)}, Charts={len(charts)}, Images={len(images_info)}"
        )

        return {
            "text_chunks": text_chunks,
            "tables": all_tables,
            "charts": charts,
            "images": images_info,
            "page_count": page_count,
        }

    # --------------------------------------------------------
    # DOCX Processing
    # --------------------------------------------------------
    def _process_docx(self, docx_path: str):
        logger.info(f"[DOCX] Processing {docx_path}")

        doc = docx.Document(docx_path)
        full_text = [p.text for p in doc.paragraphs if p.text.strip()]

        return {
            "text_chunks": [{
                "page_num": 1,
                "content": "\n".join(full_text),
                "type": "text"
            }],
            "tables": [],
            "charts": [],
            "images": [],
            "page_count": 1,
        }

    # --------------------------------------------------------
    # STORE IN PINECONE
    # --------------------------------------------------------
    def store_in_pinecone(self, processed: Dict[str, Any], doc_id: str, filename: str):

        logger.info(f"[PINECONE] Storing vectors for {doc_id}")

        # Ensure doc_id is ALWAYS string
        doc_id = str(doc_id)

        embeddings = VertexAIEmbeddings(self.embedder)

        vector_store = PineconeVectorStore(
            index=self.index,
            embedding=embeddings,
            namespace="documents"
        )

        docs = []

        # TEXT
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

        # TABLES
        for table in processed.get("tables", []):
            rows = table.get("rows", [])

            # Normalize rows (prevent NoneType errors)
            normalized_rows = []
            for row in rows:
                normalized_row = [str(cell) if cell is not None else "" for cell in row]
                normalized_rows.append(normalized_row)

            table_text = "\n".join([" | ".join(r) for r in normalized_rows])

            docs.append(Document(
                page_content=table_text,
                metadata={
                    "doc_id": doc_id,
                    "filename": filename,
                    "type": "table",
                    "page_num": table["page_num"],
                    "table_id": table["table_id"],
                    "table_name": table.get("description", "table"),
                }
            ))
        # CHARTS
        for chart in processed.get("charts", []):
            desc = f"{chart.get('description','')}\nKey Values: {chart.get('key_values','')}"
            docs.append(Document(
                page_content=desc,
                metadata={
                    "doc_id": doc_id,
                    "filename": filename,
                    "type": "chart",
                    "page_num": chart.get("page_num", 1),
                    "chart_id": chart.get("chart_id", "")
                }
            ))

        # IMAGES
        for img in processed.get("images", []):
            desc = f"{img.get('description','')}\nOCR: {img.get('text_inside','')}"
            docs.append(Document(
                page_content=desc,
                metadata={
                    "doc_id": doc_id,
                    "filename": filename,
                    "type": "image",
                    "page_num": img.get("page_num", 1),
                    "image_id": img.get("image_id", "")
                }
            ))

        vector_store.add_documents(
            documents=docs,
            ids=[str(uuid4()) for _ in docs]
        )

        logger.info(f"[PINECONE DONE] Stored {len(docs)} vectors.")



__all__ = ["GeminiMultimodalProcessor"]
