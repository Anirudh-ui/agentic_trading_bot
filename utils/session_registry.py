import redis
import uuid
import json
from custom_logging.my_logger import logger


class DocumentSessionRegistry:
    """
    Maps UI session_id → doc_id using Redis.
    Example:
        trade_doc_session_xxx  →  trade_doc_abc123
    """

    def __init__(self, host="localhost", port=6379):
        self.redis = redis.Redis(host=host, port=port, db=5, decode_responses=True)
        logger.info("[SESSION-REGISTRY] Initialized Redis mapping (DB=5)")

    # -------------------------------------------------------
    # CREATE NEW SESSION
    # -------------------------------------------------------
    def create_session(self, doc_id: str) -> str:
        """
        Create a new UI-visible session_id and link to doc_id.
        """
        session_id = f"trade_doc_session_{uuid.uuid4().hex[:12]}"
        self.redis.set(session_id, doc_id)

        logger.info(f"[SESSION-REGISTRY] Created session {session_id} → {doc_id}")
        return session_id

    # -------------------------------------------------------
    # RESOLVE SESSION → DOC_ID
    # -------------------------------------------------------
    def get_doc_id(self, session_id: str) -> str | None:
        """
        Retrieves doc_id for a given session_id.
        Returns None if not mapped.
        """
        doc_id = self.redis.get(session_id)
        logger.info(f"[SESSION-REGISTRY] Lookup: {session_id} → {doc_id}")
        return doc_id

    # -------------------------------------------------------
    # DELETE SESSION
    # -------------------------------------------------------
    def delete_session(self, session_id: str):
        self.redis.delete(session_id)
        logger.info(f"[SESSION-REGISTRY] Deleted mapping for {session_id}")

    # -------------------------------------------------------
    # LIST ALL ACTIVE SESSIONS (optional debug)
    # -------------------------------------------------------
    def list_sessions(self):
        keys = self.redis.keys("trade_doc_session_*")
        result = {k: self.redis.get(k) for k in keys}
        logger.info(f"[SESSION-REGISTRY] Active sessions: {result}")
        return result
