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

class RedisMemoryManager_document:
    """for document storage in redis (STM for doc conversations)"""

    def __init__(self, host='localhost', port=6380, db=1, ttl_hours=24):
        """
        Initialize Redis connection

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            ttl_hours: Time-to-live for session data in hours
        """
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )
        self.ttl_seconds = ttl_hours * 3600

    def _get_session_key(self, session_id: str) -> str:
        """Generate Redis key for session"""
        return f"session:{session_id}"

    def _get_messages_key(self, session_id: str) -> str:
        """Generate Redis key for messages"""
        return f"messages:{session_id}"

    def _get_metadata_key(self, session_id: str) -> str:
        """Generate Redis key for session metadata"""
        return f"metadata:{session_id}"

    def store_message(self, session_id: str, message: Dict):
        """
        Store a single message in Redis
        """
        try:
            key = self._get_messages_key(session_id)

            message['timestamp'] = datetime.now().isoformat()

            self.redis_client.rpush(key, json.dumps(message))
            self.redis_client.expire(key, self.ttl_seconds)

            self._update_session_metadata(session_id)

        except Exception as e:
            print(f"Error storing message in Redis (doc): {e}")

    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Retrieve conversation history from Redis
        """
        try:
            key = self._get_messages_key(session_id)

            if limit:
                messages = self.redis_client.lrange(key, -limit, -1)
            else:
                messages = self.redis_client.lrange(key, 0, -1)

            return [json.loads(msg) for msg in messages]

        except Exception as e:
            print(f"Error retrieving messages from Redis (doc): {e}")
            return []

    def get_langchain_messages(self, session_id: str, limit: Optional[int] = 10):
        """
        Get messages in LangChain format
        """
        messages = self.get_messages(session_id, limit)
        langchain_messages = []

        for msg in messages:
            role = msg.get('role')
            content = msg.get('content', '')

            if role == 'user':
                langchain_messages.append(HumanMessage(content=content))
            elif role == 'assistant' or role == 'bot':
                langchain_messages.append(AIMessage(content=content))
            elif role == 'system':
                langchain_messages.append(SystemMessage(content=content))

        return langchain_messages

    def _update_session_metadata(self, session_id: str):
        """Update session metadata (last activity, message count)"""
        try:
            key = self._get_metadata_key(session_id)
            messages_key = self._get_messages_key(session_id)

            metadata = {
                'last_activity': datetime.now().isoformat(),
                'message_count': self.redis_client.llen(messages_key),
                'session_id': session_id
            }

            self.redis_client.set(key, json.dumps(metadata), ex=self.ttl_seconds)

        except Exception as e:
            print(f"Error updating session metadata (doc): {e}")

    def get_session_metadata(self, session_id: str) -> Optional[Dict]:
        """Get session metadata"""
        try:
            key = self._get_metadata_key(session_id)
            data = self.redis_client.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            print(f"Error getting session metadata (doc): {e}")
            return None

    def clear_session(self, session_id: str):
        """Clear all data for a session"""
        try:
            keys = [
                self._get_messages_key(session_id),
                self._get_metadata_key(session_id),
                self._get_session_key(session_id)
            ]
            self.redis_client.delete(*keys)
        except Exception as e:
            print(f"Error clearing session (doc): {e}")

    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs"""
        try:
            pattern = "metadata:*"
            keys = self.redis_client.keys(pattern)
            return [key.split(':')[1] for key in keys]
        except Exception as e:
            print(f"Error getting active sessions (doc): {e}")
            return []
    def save_message(self, session_id: str, role: str, content: str):
        key = f"session:{session_id}:messages"
        msg = {"role": role, "content": content, "time": datetime.now().isoformat()}

        self.redis.rpush(key, json.dumps(msg))
        print(f"[REDIS][STORE] {session_id} {role}: '{content[:60]}'")

        # Run summarization + pruning check
        self.summary_buffer.process_and_prune(session_id)

    # Load STM messages (parsed)
    def load_messages(self, session_id: str):
        key = f"session:{session_id}:messages"
        raw = self.redis.lrange(key, 0, -1)
        return [json.loads(m) for m in raw]

    # Load summary (string)
    def load_summary(self, session_id: str):
        key = f"session:{session_id}:summary"
        return self.redis.get(key)

    # Clear session (on chat panel close)
    def clear_session(self, session_id: str):
        print(f"[REDIS][CLEAR] Full clear for {session_id}")
        self.redis.delete(f"session:{session_id}:messages")
        self.redis.delete(f"session:{session_id}:summary")

# ==================== WEAVIATE LTM ====================

class WeaviateMemoryManager:
    """Manages long-term memory using Weaviate"""
    
    def __init__(self, url: str, api_key: Optional[str] = None):
        """
        Initialize Weaviate connection
        
        Args:
            url: Weaviate instance URL
            api_key: Optional API key for Weaviate Cloud
        """
        if api_key:
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=url,
                auth_credentials=Auth.api_key(api_key)
            )
        else:
            #self.client = weaviate.connect_to_local(host="localhost",port=8080)
            try:
                parts = url.split("://")[-1].split(":")
                host = parts[0]
                port = int(parts[1])
                self.client = weaviate.connect_to_local(host=host, port=port)
            except Exception as e:
                # Fallback connection logic
                print(f"[WEAVIATE ERROR] Could not parse URL {url} or connect: {e}")
                self.client = weaviate.connect_to_local(host="localhost",port=8080)
        
        self._setup_schema()
    def _setup_schema(self):
        """Create Weaviate schema for conversation memory"""
        try:
            # 1. ConversationMemory Collection
            if not self.client.collections.exists("ConversationMemory"):
                
                # Correct V4 syntax for property definitions
                memory_properties = [
                    Property(name="session_id", data_type=DataType.TEXT, description="Unique session identifier"),
                    Property(name="user_id", data_type=DataType.TEXT, description="User identifier (if available)"),
                    Property(name="summary", data_type=DataType.TEXT, description="Conversation summary"),
                    Property(name="key_topics", data_type=DataType.TEXT_ARRAY, description="Main topics discussed"),
                    Property(name="user_preferences", data_type=DataType.TEXT, description="User preferences as JSON"),
                    Property(name="timestamp", data_type=DataType.DATE, description="When conversation occurred"),
                    Property(name="message_count", data_type=DataType.INT, description="Number of messages in conversation")
                ]
                
                self.client.collections.create(
                    name="ConversationMemory",
                    properties=memory_properties
                    # Vectorizer is likely set via environment variables if omitted here
                )
            
            # 2. UserProfile Collection
            if not self.client.collections.exists("UserProfile"):
                
                # Correct V4 syntax for property definitions
                profile_properties = [
                    Property(name="user_id", data_type=DataType.TEXT, description="Unique user identifier"),
                    Property(name="name", data_type=DataType.TEXT, description="User's name"),
                    Property(name="preferences", data_type=DataType.TEXT, description="User preferences as JSON"),
                    Property(name="favorite_stocks", data_type=DataType.TEXT_ARRAY, description="User's favorite stock tickers"),
                    Property(name="trading_goals", data_type=DataType.TEXT, description="User's trading goals"),
                    Property(name="last_updated", data_type=DataType.DATE, description="Last profile update")
                ]

                self.client.collections.create(
                    name="UserProfile",
                    properties=profile_properties
                )
                
        except Exception as e:
            print(f"Error setting up Weaviate schema: {e}")
    def store_conversation_summary(
        self,
        session_id: str,
        summary: str,
        key_topics: List[str],
        user_preferences: Dict,
        message_count: int,
        user_id: Optional[str] = None
    ):
        """
        Store conversation summary in Weaviate for long-term memory
        
        Args:
            session_id: Session identifier
            summary: Conversation summary
            key_topics: List of main topics discussed
            user_preferences: User preferences extracted from conversation
            message_count: Number of messages in the conversation
            user_id: Optional user identifier
        """
        now_utc = datetime.now(timezone.utc)
        timestamp_string = now_utc.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        try:
            collection = self.client.collections.get("ConversationMemory")
            
            collection.data.insert(
                properties={
                    "session_id": session_id,
                    "user_id": user_id or "anonymous",
                    "summary": summary,
                    "key_topics": key_topics,
                    "user_preferences": json.dumps(user_preferences),
                    "timestamp": timestamp_string,
                    "message_count": message_count
                }
            )
            
        except Exception as e:
            print(f"Error storing conversation summary in Weaviate: {e}")
    
    def get_user_conversation_history(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Retrieve user's conversation history from long-term memory
        
        Args:
            user_id: User identifier
            limit: Maximum number of conversations to retrieve
            
        Returns:
            List of conversation summaries
        """
        try:
            collection = self.client.collections.get("ConversationMemory")
            """{
                    "path": ["user_id"],
                    "operator": "Equal",
                    "valueText": user_id
                }"""
            response = collection.query.fetch_objects(
                filters=Filter.by_property("user_id").equal(user_id),
                limit=limit
            )
            
            results = []
            for item in response.objects:
                results.append({
                    "session_id": item.properties.get("session_id"),
                    "summary": item.properties.get("summary"),
                    "key_topics": item.properties.get("key_topics"),
                    "timestamp": item.properties.get("timestamp"),
                    "message_count": item.properties.get("message_count")
                })
            
            return results
            
        except Exception as e:
            print(f"Error retrieving conversation history from Weaviate: {e}")
            return []
    
    def search_conversations(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search through conversation summaries
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of relevant conversation summaries
        """
        try:
            collection = self.client.collections.get("ConversationMemory")
            
            response = collection.query.near_text(
                query=query,
                limit=limit
            )
            
            results = []
            for item in response.objects:
                results.append({
                    "session_id": item.properties.get("session_id"),
                    "summary": item.properties.get("summary"),
                    "key_topics": item.properties.get("key_topics"),
                    "relevance_score": item.metadata.distance
                })
            
            return results
            
        except Exception as e:
            print(f"Error searching conversations in Weaviate: {e}")
            return []
    
    def update_user_profile(
        self,
        user_id: str,
        name: Optional[str] = None,
        preferences: Optional[Dict] = None,
        favorite_stocks: Optional[List[str]] = None,
        trading_goals: Optional[str] = None
    ):
        """
        Update or create user profile
        
        Args:
            user_id: User identifier
            name: User's name
            preferences: User preferences
            favorite_stocks: List of favorite stock tickers
            trading_goals: User's trading goals
        """
        try:
            collection = self.client.collections.get("UserProfile")
            
            # Check if profile exists
            """{
                    "path": ["user_id"],
                    "operator": "Equal",
                    "valueText": user_id
                }"""
            existing = collection.query.fetch_objects(
                filters=Filter.by_property("user_id").equal(user_id),
                limit=1
            )
            
            properties = {
                "user_id": user_id,
                "last_updated": datetime.now().isoformat()
            }
            
            if name:
                properties["name"] = name
            if preferences:
                properties["preferences"] = json.dumps(preferences)
            if favorite_stocks:
                properties["favorite_stocks"] = favorite_stocks
            if trading_goals:
                properties["trading_goals"] = trading_goals
            
            if existing.objects:
                # Update existing profile
                collection.data.update(
                    uuid=existing.objects[0].uuid,
                    properties=properties
                )
            else:
                # Create new profile
                collection.data.insert(properties=properties)
                
        except Exception as e:
            print(f"Error updating user profile in Weaviate: {e}")
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get user profile from Weaviate"""
        try:
            collection = self.client.collections.get("UserProfile")
            """{
                    "path": ["user_id"],
                    "operator": "Equal",
                    "valueText": user_id
                }"""
            response = collection.query.fetch_objects(
                filters=Filter.by_property("user_id").equal(user_id),
                limit=1
            )
            
            if response.objects:
                props = response.objects[0].properties
                return {
                    "user_id": props.get("user_id"),
                    "name": props.get("name"),
                    "preferences": json.loads(props.get("preferences", "{}")),
                    "favorite_stocks": props.get("favorite_stocks", []),
                    "trading_goals": props.get("trading_goals"),
                    "last_updated": props.get("last_updated")
                }
            
            return None
            
        except Exception as e:
            print(f"Error getting user profile from Weaviate: {e}")
            return None
    
    def close(self):
        """Close Weaviate connection"""
        try:
            self.client.close()
        except Exception as e:
            print(f"Error closing Weaviate connection: {e}")
# ==================== HYBRID (DOC STM + LTM) ====================

class HybridMemoryManager:
    """
    Unified memory manager combining Redis (short-term) and Weaviate (long-term)
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

    def store_message(self, session_id: str, role: str, content: str):
        """Store message in short-term memory (Redis)"""
        message = {"role": role, "content": content}
        self.redis_manager.store_message(session_id, message)

    def get_short_term_memory(self, session_id: str, limit: int = 10):
        """Get recent conversation from Redis"""
        return self.redis_manager.get_langchain_messages(session_id, limit)

    def archive_conversation(
        self,
        session_id: str,
        summary: str,
        key_topics: List[str],
        user_preferences: Dict,
        user_id: Optional[str] = None
    ):
        """
        Archive conversation from Redis to Weaviate for long-term storage
        """
        metadata = self.redis_manager.get_session_metadata(session_id)
        message_count = metadata.get('message_count', 0) if metadata else 0

        self.weaviate_manager.store_conversation_summary(
            session_id=session_id,
            summary=summary,
            key_topics=key_topics,
            user_preferences=user_preferences,
            message_count=message_count,
            user_id=user_id
        )

    def get_long_term_context(self, user_id: str, limit: int = 5) -> str:
        """
        Get long-term conversation context for a user
        """
        history = self.weaviate_manager.get_user_conversation_history(user_id, limit)

        if not history:
            return "No previous conversation history."

        context = "Previous conversations:\n"
        for conv in history:
            context += f"- {conv['summary']} (Topics: {', '.join(conv['key_topics'])})\n"

        return context

    def close(self):
        """Close all connections"""
        self.weaviate_manager.close()
