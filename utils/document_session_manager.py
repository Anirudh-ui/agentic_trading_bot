"""
Document Session Manager (FINAL VERSION)
Handles:
- session_id → doc_id mapping
- calls HybridMemoryManager for STM+LTM
"""

from typing import Optional, List, Dict
from custom_logging.my_logger import logger


class DocumentSessionRegistry:
    """Maps session_id → doc_id."""

    def __init__(self):
        self.sessions = {}

    def create_session(self, doc_id: str) -> str:
        import uuid
        session_id = f"trade_doc_session_{uuid.uuid4().hex[:12]}"
        self.sessions[session_id] = doc_id

        logger.info(f"[SESSION-REGISTRY] Created session {session_id} → {doc_id}")
        return session_id

    def get_doc_id(self, session_id: str) -> Optional[str]:
        doc_id = self.sessions.get(session_id)
        logger.info(f"[SESSION-REGISTRY] Lookup: {session_id} → {doc_id}")
        return doc_id


class DocumentSessionManager:
    """High-level wrapper that exposes LTM + STM API to FastAPI routes."""

    def __init__(self, memory_manager):
        self.memory = memory_manager

    # -------------------- STORE QA --------------------
    def store_document_qa(self, doc_id: str, user_id: str,
                          question: str, answer: str, sources: List):
        try:
            self.memory.store_qa(doc_id, user_id, question, answer, sources)
        except Exception as e:
            logger.error(f"[STORE Q&A ERROR] {e}")

    # -------------------- SUMMARIES --------------------
    def get_document_summary(self, doc_id: str):
        stm = self.memory.get_summary(doc_id)
        if stm:
            return stm

        ltm = self.memory.get_ltm_summary(doc_id)
        return ltm or ""

    # -------------------- HISTORY --------------------
    def get_last_document_qa(self, doc_id: str, limit=3):
        try:
            return self.memory.get_last_qa(doc_id, limit)
        except Exception as e:
            logger.error(f"[GET LAST Q&A ERROR] {e}")
            return []

    def get_document_qa_count(self, doc_id: str):
        try:
            return self.memory.get_qa_count(doc_id)
        except Exception as e:
            logger.error(f"[GET QA COUNT ERROR] {e}")
            return 0

    # -------------------- HISTORY PAYLOAD --------------------
    def build_document_history_payload(self, doc_id: str):
        summary = self.get_document_summary(doc_id)
        last_qa = self.get_last_document_qa(doc_id)

        return {
            "has_history": bool(summary or last_qa),
            "summary": summary,
            "last_qa": last_qa
        }

    # -------------------- DELETE --------------------
    def delete_document_history(self, doc_id: str):
        try:
            self.memory.delete_document(doc_id)
        except Exception as e:
            logger.error(f"[DELETE HISTORY ERROR] {e}")
