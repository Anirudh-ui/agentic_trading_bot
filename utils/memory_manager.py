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
from typing import List, Dict, Optional
from utils.model_loaders import ModelLoader
from utils.gemini_ner import extract_entities_with_gemini
import regex as re
from custom_logging.my_logger import logger
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
class DocumentConversationSummaryBuffer:
    """
    Document-only conversation summarizer.
    Keeps rolling conversation summary (<100 words).
    """

    def __init__(self, llm, min_messages=4, keep_last=3):
        self.llm = llm
        self.min_messages = min_messages
        self.keep_last = keep_last

    def should_summarize(self, messages: List[Dict]) -> bool:
        """Trigger summarization when >= min_messages since last summary."""
        if len(messages) < self.min_messages:
            return False

        count_since = 0
        for m in reversed(messages):
            if m["role"] == "summary":
                break
            count_since += 1

        return count_since >= self.min_messages

    def summarize(self, messages: List[Dict], current_summary: str) -> Dict:
        """
        Produce new summary + return pruned STM messages.
        """

        llm_prompt = f"""
You are updating a rolling conversation summary (<100 words).
Keep it factual and tied to the document discussion.

Current Summary:
{current_summary}

Recent Messages:
"""

        tail = messages[-self.min_messages:]
        formatted_tail = "\n".join(f"{m['role']}: {m['content']}" for m in tail)
        prompt = llm_prompt + formatted_tail

        resp = self.llm.invoke([HumanMessage(content=prompt)])
        new_summary = resp.content.strip()

        summary_msg = {
            "role": "summary",
            "content": new_summary,
            "timestamp": datetime.now().isoformat()
        }

        trimmed = [summary_msg] + messages[-self.keep_last:]

        return {"summary": new_summary, "trimmed_messages": trimmed}


# ===================================================================
# 2) DOCUMENT STM → REDIS
# ===================================================================
class RedisMemoryManager_document:
    """Short-term memory for a document (per doc_id). Handles:
       - message storage
       - rolling summary storage
       - summarization trigger
    """

    def __init__(self, host="localhost", port=6380, db=1, ttl_hours=48):
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.ttl = ttl_hours * 3600

        # Groq or Gemini LLM (via ModelLoader)
        self.llm = ModelLoader().load_llm()

    def _msg_key(self, doc_id):
        return f"doc_msgs:{doc_id}"

    def _summary_key(self, doc_id):
        return f"doc_summary:{doc_id}"

    # ----------------------------------------
    # STORE MESSAGE
    # ----------------------------------------
    def store_message(self, doc_id: str, message: Dict):
        message["timestamp"] = datetime.now().isoformat()

        self.redis.rpush(self._msg_key(doc_id), json.dumps(message))
        self.redis.expire(self._msg_key(doc_id), self.ttl)

    # ----------------------------------------
    # GET MESSAGES
    # ----------------------------------------
    def get_messages(self, doc_id: str, limit=None) -> List[Dict]:
        key = self._msg_key(doc_id)
        if limit:
            raw = self.redis.lrange(key, -limit, -1)
        else:
            raw = self.redis.lrange(key, 0, -1)
        return [json.loads(x) for x in raw]

    # ----------------------------------------
    # SUMMARY
    # ----------------------------------------
    def get_summary(self, doc_id: str) -> Optional[str]:
        return self.redis.get(self._summary_key(doc_id))

    def store_summary(self, doc_id: str, summary: str):
        self.redis.set(self._summary_key(doc_id), summary, ex=self.ttl)

    # ----------------------------------------
    # CLEAR
    # ----------------------------------------
    def clear_session(self, doc_id: str):
        self.redis.delete(self._msg_key(doc_id))
        self.redis.delete(self._summary_key(doc_id))


# ===============================================================
# 2) DOCUMENT LTM (Weaviate)
# ===============================================================

class WeaviateMemoryManager:
    """Persistent Q&A + Summaries stored per doc."""

    def __init__(self, url, api_key=None):
        try:
            if api_key:
                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=url,
                    auth_credentials=Auth.api_key(api_key)
                )
            else:
                self.client = weaviate.connect_to_local()

            self._setup_schema()
            logger.info("[WEAVIATE] Connected.")

        except Exception as e:
            logger.error(f"[WEAVIATE INIT ERROR] {e}")
            self.client = None

    # -------------------------------------------------------
    # SCHEMA
    # -------------------------------------------------------
    def _setup_schema(self):
        if not self.client:
            return

        cols = self.client.collections

        if not cols.exists("DocumentQA"):
            cols.create(
                name="DocumentQA",
                properties=[
                    Property(name="doc_id", data_type=DataType.TEXT),
                    Property(name="user_id", data_type=DataType.TEXT),
                    Property(name="question", data_type=DataType.TEXT),
                    Property(name="answer", data_type=DataType.TEXT),
                    Property(name="sources", data_type=DataType.TEXT),
                    Property(name="timestamp", data_type=DataType.DATE),
                ]
            )

        if not cols.exists("DocumentSummary"):
            cols.create(
                name="DocumentSummary",
                properties=[
                    Property(name="doc_id", data_type=DataType.TEXT),
                    Property(name="summary", data_type=DataType.TEXT),
                    Property(name="timestamp", data_type=DataType.DATE),
                ]
            )

    # -------------------------------------------------------
    # Q&A STORAGE
    # -------------------------------------------------------
    def store_document_qa(self, doc_id, user_id, question, answer, sources):
        try:
            self.client.collections.get("DocumentQA").data.insert(
                properties={
                    "doc_id": doc_id,
                    "user_id": user_id,
                    "question": question,
                    "answer": answer,
                    "sources": json.dumps(sources or []),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        except Exception as e:
            logger.error(f"[WEAVIATE QA ERROR] {e}")

    def get_last_document_qa(self, doc_id, limit=3):
        try:
            res = self.client.collections.get("DocumentQA").query.fetch_objects(
                filters=Filter.by_property("doc_id").equal(doc_id),
                limit=limit
            )

            out = []
            for o in res.objects or []:
                p = o.properties
                out.append({
                    "question": p.get("question", ""),
                    "answer": p.get("answer", "")
                })
            return out

        except Exception as e:
            logger.error(f"[WEAVIATE GET QA ERROR] {e}")
            return []

    # -------------------------------------------------------
    # SUMMARY STORAGE
    # -------------------------------------------------------
    def store_summary(self, doc_id, summary):
        try:
            self.client.collections.get("DocumentSummary").data.insert(
                properties={
                    "doc_id": doc_id,
                    "summary": summary,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        except Exception as e:
            logger.error(f"[WEAVIATE SUMMARY ERROR] {e}")

    def get_document_summary(self, doc_id):
        try:
            res = self.client.collections.get("DocumentSummary").query.fetch_objects(
                filters=Filter.by_property("doc_id").equal(doc_id),
                limit=1
            )
            if not res.objects:
                return ""
            return res.objects[0].properties.get("summary", "")
        except Exception as e:
            logger.error(f"[WEAVIATE GET SUMMARY ERROR] {e}")
            return ""

    # -------------------------------------------------------
    # Q&A COUNT (AGGREGATE)
    # -------------------------------------------------------
    def get_qa_count(self, doc_id: str) -> int:
        try:
            res = self.client.collections.get("DocumentQA").aggregate.over_all(
                filters=Filter.by_property("doc_id").equal(doc_id)
            )
            return res.total_count or 0
        except Exception as e:
            logger.error(f"[WEAVIATE QA COUNT ERROR] {e}")
            return 0

    # -------------------------------------------------------
    # DELETE DOCUMENT
    # -------------------------------------------------------
    def delete_document(self, doc_id):
        try:
            self.client.collections.get("DocumentQA").data.delete_many(
                Filter.by_property("doc_id").equal(doc_id)
            )
            self.client.collections.get("DocumentSummary").data.delete_many(
                Filter.by_property("doc_id").equal(doc_id)
            )
        except Exception as e:
            logger.error(f"[WEAVIATE DELETE ERROR] {e}")


# ===============================================================
# 3) HYBRID MEMORY (STM + LTM)
# ===============================================================

class HybridMemoryManager:

    def __init__(self, redis_host, redis_port,
                 weaviate_url=None, weaviate_api_key=None):

        self.stm = RedisMemoryManager_document(host=redis_host, port=redis_port)
        self.weaviate = None

        if weaviate_url:
            try:
                self.weaviate = WeaviateMemoryManager(weaviate_url, weaviate_api_key)
            except:
                logger.error("[HYBRID] Weaviate LTM disabled.")

    # -------------- STM ----------------
    def store_message(self, doc_id, role, content):
        self.stm.store_message(doc_id, {"role": role, "content": content})

    def get_short_term_memory(self, doc_id, limit=5):
        return self.stm.get_messages(doc_id, limit)

    def get_summary(self, doc_id):
        return self.stm.get_summary(doc_id)

    # -------------- LTM ----------------
    def store_qa(self, doc_id, user_id, question, answer, sources):
        if self.weaviate:
            self.weaviate.store_document_qa(doc_id, user_id, question, answer, sources)

    def store_summary(self, doc_id, summary):
        if self.weaviate:
            self.weaviate.store_summary(doc_id, summary)

    def get_last_qa(self, doc_id, limit=3):
        if self.weaviate:
            return self.weaviate.get_last_document_qa(doc_id, limit)
        return []

    def get_ltm_summary(self, doc_id):
        if self.weaviate:
            return self.weaviate.get_document_summary(doc_id)
        return None

    def get_qa_count(self, doc_id):
        if self.weaviate:
            return self.weaviate.get_qa_count(doc_id)
        return 0

    # -------------- DELETE ----------------
    def delete_document(self, doc_id):
        self.stm.clear_session(doc_id)
        if self.weaviate:
            self.weaviate.delete_document(doc_id)

    def close(self):
        try:
            if self.weaviate and self.weaviate.client:
                self.weaviate.client.close()
        except:
            pass
