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
    Short-term memory for document chats.
    Includes:
    - Message storage (Redis)
    - Automatic summarization using ConversationSummaryBuffer
    - Metadata tracking
    """

    def __init__(self, host='localhost', port=6380, db=1, ttl_hours=24):
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )
        self.ttl_seconds = ttl_hours * 3600

        # Create summarizer (REQUIRED)
        llm = ModelLoader().load_llm()
        self.summary_buffer = ConversationSummaryBuffer(llm)

    # ------------------ Redis Key Generators ------------------
    
    def _get_messages_key(self, doc_id: str) -> str:
        return f"messages:{doc_id}"

    def _get_summary_key(self, doc_id: str) -> str:
        return f"summary:{doc_id}"

    def _get_metadata_key(self, doc_id: str) -> str:
        return f"metadata:{doc_id}"

    # ------------------ Store Message ------------------

    def store_message(self, doc_id: str, message: Dict):
        """
        Store a message in Redis STM.
        Auto-summarize when needed.
        """
        try:
            key = self._get_messages_key(doc_id)
            message["timestamp"] = datetime.now().isoformat()

            # Save raw message
            self.redis_client.rpush(key, json.dumps(message))
            self.redis_client.expire(key, self.ttl_seconds)

            # Update metadata
            self._update_metadata(doc_id)

            # Attempt summarization
            self._maybe_summarize_and_prune(doc_id)

        except Exception as e:
            print(f"[REDIS_DOC] Error storing message: {e}")

    # ------------------ Summarization Logic ------------------

    def _maybe_summarize_and_prune(self, doc_id: str):
        """
        Summarize last N messages and prune excess.
        Stores the summary separately under summary:{doc_id}.
        """
        try:
            messages = self.get_messages(doc_id)

            # Let buffer decide if summary is needed
            if not self.summary_buffer.should_summarize(messages):
                return

            result = self.summary_buffer.summarize_and_prune(messages)

            # Save the summary
            summary_text = result["summary"]
            self.redis_client.set(self._get_summary_key(doc_id), summary_text, ex=self.ttl_seconds)

            # Replace messages with trimmed set
            key = self._get_messages_key(doc_id)
            self.redis_client.delete(key)

            for msg in result["trimmed_messages"]:
                self.redis_client.rpush(key, json.dumps(msg))

            self.redis_client.expire(key, self.ttl_seconds)

        except Exception as e:
            print(f"[REDIS_DOC] Error summarizing/pruning: {e}")

    # ------------------ Retrieval ------------------

    def get_messages(self, doc_id: str, limit: Optional[int] = None) -> List[Dict]:
        try:
            key = self._get_messages_key(doc_id)
            raw = self.redis_client.lrange(key, -limit, -1) if limit else self.redis_client.lrange(key, 0, -1)
            return [json.loads(m) for m in raw]
        except Exception as e:
            print(f"[REDIS_DOC] Error retrieving messages: {e}")
            return []

    def get_langchain_messages(self, doc_id: str, limit: int = 10):
        msgs = self.get_messages(doc_id, limit)
        out = []

        for m in msgs:
            role = m.get("role")
            content = m.get("content", "")

            if role == "user":
                out.append(HumanMessage(content=content))
            elif role == "assistant" or role == "bot":
                out.append(AIMessage(content=content))
            elif role == "summary":
                out.append(SystemMessage(content=f"(Summary) {content}"))

        return out

    def get_summary(self, doc_id: str):
        return self.redis_client.get(self._get_summary_key(doc_id))

    # ------------------ Metadata ------------------

    def _update_metadata(self, doc_id):
        try:
            key = self._get_metadata_key(doc_id)
            messages_key = self._get_messages_key(doc_id)

            metadata = {
                "last_activity": datetime.now().isoformat(),
                "message_count": self.redis_client.llen(messages_key),
                "doc_id": doc_id
            }

            self.redis_client.set(key, json.dumps(metadata), ex=self.ttl_seconds)

        except Exception as e:
            print(f"[REDIS_DOC] Error updating metadata: {e}")

    def get_session_metadata(self, doc_id: str) -> Optional[Dict]:
        try:
            meta = self.redis_client.get(self._get_metadata_key(doc_id))
            return json.loads(meta) if meta else None
        except:
            return None

    # ------------------ Clear Session ------------------

    def clear_session(self, doc_id: str):
        try:
            self.redis_client.delete(self._get_messages_key(doc_id))
            self.redis_client.delete(self._get_summary_key(doc_id))
            self.redis_client.delete(self._get_metadata_key(doc_id))
        except Exception as e:
            print(f"[REDIS_DOC] Error clearing session: {e}")


# ==================== WEAVIATE (DOCUMENT LTM USING doc_id) ====================

class WeaviateMemoryManager:
    """Long-term document memory manager (patched to prevent socket leaks)."""

    def __init__(self, url: str, api_key: Optional[str] = None):
        self.client = None
        self.url = url
        self.api_key = api_key

        try:
            # Cloud connection
            if api_key:
                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=url,
                    auth_credentials=Auth.api_key(api_key)
                )
            else:
                # Local connection fallback
                try:
                    parsed = url.split("://")[-1].split(":")
                    host = parsed[0]
                    port = int(parsed[1])
                except:
                    host, port = "localhost", 8080

                # Local client requires grpc_port
                self.client = weaviate.connect_to_local(
                    host=host,
                    port=port,
                    grpc_port=50051,
                    embedded=False
                )

            self._setup_schema()

        except Exception as e:
            print(f"[WEAVIATE] Failed to connect: {e}")
            self.client = None

    def _setup_schema(self):
        """Ensure ConversationMemory schema exists."""
        if not self.client:
            return

        try:
            if not self.client.collections.exists("ConversationMemory"):
                self.client.collections.create(
                    name="ConversationMemory",
                    properties=[
                        Property(name="doc_id", data_type=DataType.TEXT),
                        Property(name="summary", data_type=DataType.TEXT),
                        Property(name="key_topics", data_type=DataType.TEXT_ARRAY),
                        Property(name="timestamp", data_type=DataType.DATE),
                        Property(name="message_count", data_type=DataType.INT),
                    ]
                )
        except Exception as e:
            print(f"[WEAVIATE] Schema setup error: {e}")

    def store_conversation_summary(
        self,
        doc_id: str,
        summary: str,
        key_topics: List[str],
        message_count: int
    ):
        """Store summary in Weaviate LTM."""
        if not self.client:
            print("[WEAVIATE] No active client. Skipping insert.")
            return

        timestamp_string = (
            datetime.now(timezone.utc)
            .strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + "Z"
        )

        try:
            self.client.collections.get("ConversationMemory").data.insert(
                properties={
                    "doc_id": doc_id,
                    "summary": summary,
                    "key_topics": key_topics,
                    "timestamp": timestamp_string,
                    "message_count": message_count
                }
            )
        except Exception as e:
            print(f"[WEAVIATE] Error storing summary: {e}")

    def close(self):
        """Safely close Weaviate client connection."""
        try:
            if self.client:
                self.client.close()
        except:
            pass
        finally:
            self.client = None


# ==================== HYBRID MEMORY (DOCUMENT STM + LTM) ====================

class HybridMemoryManager:
    """
    Combines:
    Redis (STM)
    Weaviate (LTM)
    """

    def __init__(
        self,
        redis_host='localhost',
        redis_port=6380,
        weaviate_url='http://localhost:8080',
        weaviate_api_key=None,
        session_ttl_hours=24
    ):
        self.redis_manager = RedisMemoryManager_document(
            host=redis_host,
            port=redis_port,
            ttl_hours=session_ttl_hours
        )
        self.weaviate_manager = WeaviateMemoryManager(
            url=weaviate_url,
            api_key=weaviate_api_key
        )

    # ------------------ STM ------------------

    def store_message(self, doc_id: str, role: str, content: str):
        msg = {"role": role, "content": content}
        self.redis_manager.store_message(doc_id, msg)

    def get_short_term_memory(self, doc_id: str, limit: int = 10):
        return self.redis_manager.get_langchain_messages(doc_id, limit)

    # ------------------ LTM ------------------

    def archive_conversation(
        self,
        doc_id: str,
        summary: str,
        key_topics: List[str]
    ):
        metadata = self.redis_manager.get_session_metadata(doc_id)
        message_count = metadata.get("message_count", 0) if metadata else 0

        self.weaviate_manager.store_conversation_summary(
            doc_id=doc_id,
            summary=summary,
            key_topics=key_topics,
            message_count=message_count
        )

    def close(self):
        self.weaviate_manager.close()

