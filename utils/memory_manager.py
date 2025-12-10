"""
Memory Management System for Trading Chatbot
- Redis: Short-term memory (conversation history, session data)
- Weaviate: Long-term memory (persistent conversation summaries, user preferences)
- Pinecone: Document knowledge base (unchanged)
"""

import redis
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType
from weaviate.classes.query import Filter

from typing import List, Dict, Optional
import json
from datetime import datetime, timedelta, timezone
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import hashlib
import os
import sys
from dotenv import load_dotenv
import json
import time
from custom_logging import logger
from typing import List, Dict, Optional
from utils.model_loaders import ModelLoader
from utils.gemini_ner import extract_entities_with_gemini
import regex as re
from langchain_core.messages import HumanMessage , AIMessage , SystemMessage
load_dotenv()



# ==================== CONVERSATION SUMMARY BUFFER ====================

class ConversationSummaryBuffer:
    """
    Summarizer for session-level STM.
    Does NOT summarize user-level entity memory.
    """

    def __init__(self, llm, min_messages_to_summarize=6, lookback_messages=10, keep_last_messages=3):
        self.llm = llm
        self.min_messages_to_summarize = min_messages_to_summarize
        self.lookback_messages = lookback_messages
        self.keep_last_messages = keep_last_messages

    def _extract_existing_summary(self, messages: List[Dict]) -> str:
        for m in reversed(messages):
            if m.get("role") == "summary":
                return m.get("content", "")
        return ""

    def _build_summary_prompt(self, recent_messages: List[Dict], existing_summary: str) -> str:
        conv_text = ""
        for m in recent_messages:
            conv_text += f"{m.get('role').upper()}: {m.get('content')}\n"

        return f"""
You are a summarization assistant.
Write a factual 3-5 sentence summary of the conversation.
Do NOT hallucinate.
Do NOT include tool dumps.
Do NOT include speculation.
Use only what is explicitly said.

Existing summary:
{existing_summary}

Conversation:
{conv_text}

Return plaintext only.
"""

    def should_summarize(self, messages: List[Dict]) -> bool:
        if len(messages) < self.min_messages_to_summarize:
            return False

        count_since_summary = 0
        for m in reversed(messages):
            if m.get("role") == "summary":
                break
            count_since_summary += 1

        return count_since_summary >= self.min_messages_to_summarize

    def summarize_and_prune(self, messages: List[Dict]) -> Dict[str, any]:
        recent_messages = messages[-self.lookback_messages:]
        existing = self._extract_existing_summary(messages)

        prompt = self._build_summary_prompt(recent_messages, existing)
        result = self.llm.invoke([HumanMessage(content=prompt)])
        summary_text = result.content.strip()

        summary_message = {
            "role": "summary",
            "content": summary_text,
            "timestamp": datetime.now().isoformat()
        }

        tail = messages[-self.keep_last_messages:]
        trimmed = [summary_message] + tail

        return {
            "summary": summary_text,
            "trimmed_messages": trimmed
        }


class RedisMemoryManager:
    """
    SHORT-TERM MEMORY for each SESSION
    + USER-LEVEL PERSISTENT ENTITY MEMORY
    """

    def __init__(self, host="localhost", port=6380, db=0, ttl_hours=2):
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.ttl_seconds = ttl_hours * 3600

        # Summarization configuration
        self.min_messages_to_summarize = 6
        self.lookback_messages = 10
        self.keep_last_messages = 3

    # ------------------ REDIS KEYS ------------------
    def _session_messages_key(self, session_id):
        return f"messages:{session_id}"

    def _session_metadata_key(self, session_id):
        return f"metadata:{session_id}"

    def _user_entity_key(self, user_id):
        return f"entity:{user_id}"

    # ------------------ STORE ENTITY MEMORY (USER-LEVEL) ------------------
    def store_entity(self, user_id: str, field: str, value: str):
        if value is None:
            return

        # If it's a list or dict, JSON encode it
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False)
        else:
            value = str(value)

        self.redis.hset(self._user_entity_key(user_id), field, value)

    def get_entity(self, user_id: str, field: str) -> Optional[str]:
        return self.redis.hget(self._user_entity_key(user_id), field)

    def get_all_entities(self, user_id: str) -> Dict[str, str]:
        return self.redis.hgetall(self._user_entity_key(user_id))

    # ------------------ ENTITY EXTRACTION ------------------
    def extract_entities(self, user_id: str, text: str):
            """
            Uses Gemini NER for robust entity extraction.
            Falls back to simple rules if needed.
            """

            ner = extract_entities_with_gemini(text)

            # Store Gemini-detected entities
            if ner.get("person_name"):
                self.store_entity(user_id, "user_name", ner["person_name"])

            if ner.get("location"):
                self.store_entity(user_id, "user_location", ner["location"])

            if ner.get("company"):
                self.store_entity(user_id, "last_company", ner["company"])

            if ner.get("ticker"):
                self.store_entity(user_id, "last_ticker", ner["ticker"])

            # Store keywords (optional)
            if ner.get("other_keywords"):
                self.store_entity(user_id, "keywords", ner["other_keywords"])

    # ------------------ STORE SESSION MESSAGE ------------------
    def store_message(self, session_id: str, message: Dict):
        key = self._session_messages_key(session_id)
        message["timestamp"] = datetime.now().isoformat()
        self.redis.rpush(key, json.dumps(message))
        self.redis.expire(key, self.ttl_seconds)
        self._update_metadata(session_id)
        self._maybe_summarize_and_prune(session_id)

    # ------------------ RETRIEVE ------------------
    def get_messages(self, session_id, limit=None):
        key = self._session_messages_key(session_id)
        msgs = self.redis.lrange(key, 0, -1) if limit is None else self.redis.lrange(key, -limit, -1)
        return [json.loads(x) for x in msgs]

    def get_langchain_messages(self, session_id, limit=10):
        msgs = self.get_messages(session_id, limit)
        out = []
        for m in msgs:
            role = m["role"]
            content = m["content"]
            if role == "user":
                out.append(HumanMessage(content=content))
            elif role == "assistant":
                out.append(AIMessage(content=content))
            elif role == "summary":
                out.append(SystemMessage(content=f"(Summary) {content}"))
        return out

    # ------------------ SESSION META ------------------
    def _update_metadata(self, session_id):
        key = self._session_metadata_key(session_id)
        msgs_key = self._session_messages_key(session_id)
        meta = {
            "last_activity": datetime.now().isoformat(),
            "message_count": self.redis.llen(msgs_key),
            "session_id": session_id
        }
        self.redis.set(key, json.dumps(meta), ex=self.ttl_seconds)

    # ------------------ CLEAR SESSION ------------------
    def clear_session(self, session_id):
        self.redis.delete(self._session_messages_key(session_id))
        self.redis.delete(self._session_metadata_key(session_id))

    # ------------------ SUMMARIZATION LOGIC ------------------
    def _maybe_summarize_and_prune(self, session_id):
        messages = self.get_messages(session_id)
        if len(messages) < self.min_messages_to_summarize:
            return
        if messages[-1]["role"] == "summary":
            return

        llm = ModelLoader().load_llm()
        buffer = ConversationSummaryBuffer(llm, self.min_messages_to_summarize, self.lookback_messages, self.keep_last_messages)

        if not buffer.should_summarize(messages):
            return

        result = buffer.summarize_and_prune(messages)
        trimmed = result["trimmed_messages"]

        key = self._session_messages_key(session_id)
        self.redis.delete(key)
        for m in trimmed:
            self.redis.rpush(key, json.dumps(m))

        self.redis.expire(key, self.ttl_seconds)

# ==================== REDIS (DOCUMENT STM) ====================
# ==================== REDIS (DOCUMENT STM WITH SUMMARIZATION) ====================

class RedisMemoryManager_document:
    """
    Short-term memory for document chat:
    - Stores raw messages
    - Prunes to keep last N
    - Automatic summarization
    """

    def __init__(self, host='localhost', port=6380, db=1, ttl_hours=48):
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.ttl_seconds = ttl_hours * 3600
        self.llm = ModelLoader().load_llm()
        self.summary_buffer = ConversationSummaryBuffer(self.llm)

    def _key(self, doc_id): return f"messages:{doc_id}"
    def _summary_key(self, doc_id): return f"summary:{doc_id}"

    def store_message(self, doc_id: str, message: dict):
        key = self._key(doc_id)
        message["timestamp"] = datetime.now().isoformat()

        self.redis.rpush(key, json.dumps(message))
        self.redis.expire(key, self.ttl_seconds)

        self._maybe_summarize_and_prune(doc_id)

    def _maybe_summarize_and_prune(self, doc_id):
        msgs = self.get_messages(doc_id)

        if not self.summary_buffer.should_summarize(msgs):
            return

        result = self.summary_buffer.summarize_and_prune(msgs)

        # store rolling summary
        self.redis.set(self._summary_key(doc_id), result["summary"], ex=self.ttl_seconds)

        # rewrite STM memory
        self.redis.delete(self._key(doc_id))
        for m in result["trimmed_messages"]:
            self.redis.rpush(self._key(doc_id), json.dumps(m))

    def get_messages(self, doc_id, limit=None):
        raw = self.redis.lrange(self._key(doc_id), -limit, -1) if limit else \
              self.redis.lrange(self._key(doc_id), 0, -1)
        return [json.loads(r) for r in raw]

    def get_summary(self, doc_id):
        return self.redis.get(self._summary_key(doc_id))

    def clear_session(self, doc_id):
        self.redis.delete(self._key(doc_id))
        self.redis.delete(self._summary_key(doc_id))

# ==================== WEAVIATE (DOCUMENT LTM USING doc_id) ====================

class WeaviateMemoryManager:
    """Long-term memory for document Q&A + rolling summary."""

    def __init__(self, url, api_key=None):
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=Auth.api_key(api_key)
        ) if api_key else \
        weaviate.connect_to_local()

        self._setup_schema()

    def _setup_schema(self):
        c = self.client.collections

        if not c.exists("DocumentQA"):
            c.create(
                name="DocumentQA",
                properties=[
                    Property(name="doc_id", data_type=DataType.TEXT),
                    Property(name="user_id", data_type=DataType.TEXT),
                    Property(name="question", data_type=DataType.TEXT),
                    Property(name="answer", data_type=DataType.TEXT),
                    Property(name="sources", data_type=DataType.TEXT),
                    Property(name="timestamp", data_type=DataType.DATE),

                    # NEW
                    Property(name="summary", data_type=DataType.TEXT),
                    Property(name="topics", data_type=DataType.TEXT_ARRAY),
                ]
            )

    def store_qa(self, doc_id, user_id, question, answer, sources):
        now = datetime.now(timezone.utc).isoformat()

        self.client.collections.get("DocumentQA").data.insert(
            properties={
                "doc_id": doc_id,
                "user_id": user_id,
                "question": question,
                "answer": answer,
                "sources": json.dumps(sources),
                "timestamp": now,
            }
        )

    def store_summary(self, doc_id, summary, topics):
        now = datetime.now(timezone.utc).isoformat()

        self.client.collections.get("DocumentQA").data.insert(
            properties={
                "doc_id": doc_id,
                "summary": summary,
                "topics": topics,
                "timestamp": now
            }
        )

    def get_last_qa(self, doc_id, limit=3):
        res = self.client.query.get(
            "DocumentQA",
            ["question", "answer", "timestamp"]
        ).with_where({
            "path": ["doc_id"], "operator": "Equal", "valueString": doc_id
        }).with_sort([
            {"path": ["timestamp"], "order": "desc"}
        ]).with_limit(limit).do()

        items = res["data"]["Get"]["DocumentQA"]
        return [{"question": x["question"], "answer": x["answer"]} for x in items if x.get("question")]

    def get_summary(self, doc_id):
        res = self.client.query.get(
            "DocumentQA",
            ["summary", "timestamp"]
        ).with_where({
            "path": ["doc_id"], "operator": "Equal", "valueString": doc_id
        }).with_sort([
            {"path": ["timestamp"], "order": "desc"}
        ]).with_limit(1).do()

        items = res["data"]["Get"]["DocumentQA"]
        if items and items[0].get("summary"):
            return items[0]["summary"]
        return None


# ==================== HYBRID MEMORY (DOCUMENT STM + LTM) ====================
class HybridMemoryManager:
    """
    Unified STM (Redis) + LTM (Weaviate) memory manager.
    Fully backward-compatible with both old and new constructor signatures.
    """

    def __init__(
        self,
        redis_host='localhost',
        redis_port=6379,
        weaviate_url=None,
        weaviate_api_key=None,
        session_ttl_hours=48,        # <-- added for backward compatibility
        **kwargs                     # <-- absorbs any unexpected args
    ):
        """
        Accepts both old and new parameters safely.
        """
        try:
            # STM (Redis)
            self.stm = RedisMemoryManager_document(
                host=redis_host,
                port=redis_port,
                ttl_hours=session_ttl_hours
            )

            # LTM (Weaviate)
            if weaviate_url:
                self.weaviate_manager = WeaviateMemoryManager(
                    url=weaviate_url,
                    api_key=weaviate_api_key
                )
            else:
                # If URL missing → don't break startup
                logger.warning("[HYBRID MEMORY] Weaviate disabled (no URL provided)")
                self.weaviate_manager = None

            logger.info("[HYBRID MEMORY] Initialized successfully")

        except Exception as e:
            logger.error(f"[HYBRID MEMORY] Initialization failed: {e}")
            raise

    # ============================================================
    # STM (Redis)
    # ============================================================

    def store_message(self, doc_id: str, role: str, content: str):
        self.stm.store_message(doc_id, {"role": role, "content": content})

    def get_short_term_memory(self, doc_id: str, limit: int = 5):
        return self.stm.get_messages(doc_id, limit)

    def get_summary(self, doc_id: str):
        return self.stm.get_summary(doc_id)

    def clear(self, doc_id: str):
        self.stm.clear_session(doc_id)

    # ============================================================
    # LTM (Weaviate)
    # ============================================================

    def store_qa(self, doc_id, user_id, question, answer, sources):
        if not self.weaviate_manager:
            return
        self.weaviate_manager.store_document_qa(
            doc_id=doc_id,
            user_id=user_id,
            question=question,
            answer=answer,
            sources=sources
        )

    def store_summary(self, doc_id, summary, topics=["general"]):
        if not self.weaviate_manager:
            return
        self.weaviate_manager.store_summary(
            doc_id=doc_id,
            summary=summary,
            topics=topics
        )

    def get_last_qa(self, doc_id, limit=3):
        if not self.weaviate_manager:
            return []
        return self.weaviate_manager.get_last_document_qa(doc_id, limit)

    def get_ltm_summary(self, doc_id):
        if not self.weaviate_manager:
            return None
        return self.weaviate_manager.get_document_summary(doc_id)

