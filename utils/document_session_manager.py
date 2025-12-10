"""
Enhanced Document Session Manager
- Document registration and metadata management
- Session tracking (STM + LTM)
- Conversation history retrieval
- Duplicate document detection
- Comprehensive logging and error handling
"""

import json
from typing import List, Dict, Optional
from datetime import datetime, timezone
import hashlib
import sys

from utils.memory_manager import RedisMemoryManager_document, WeaviateMemoryManager
from weaviate.classes.config import Property, DataType
from weaviate.classes.query import Filter
from custom_logging.my_logger import logger, log_execution_time
from exception.exceptions import (
    TradingBotException,
    SessionException,
    ValidationException
)


class DocumentSessionManager:
    """
    Manages document sessions with enhanced features:
    - Document metadata (Weaviate LTM – legacy, Postgres is now source of truth)
    - Session management (Redis STM)
    - Q&A history (Weaviate LTM)
    - Duplicate prevention
    - Conversation summaries
    """
    
    def __init__(
        self,
        redis_host: str = 'localhost',
        redis_port: int = 6380,
        weaviate_url: str = 'http://localhost:8080',
        weaviate_api_key: Optional[str] = None
    ):
        """
        Initialize document session manager
        
        Args:
            redis_host: Redis host for STM
            redis_port: Redis port
            weaviate_url: Weaviate URL for LTM
            weaviate_api_key: Optional Weaviate API key
        """
        try:
            logger.info("[SESSION MANAGER] Initializing...")
            
            # Redis for short-term memory (document STM)
            self.redis_manager = RedisMemoryManager_document(
                host=redis_host,
                port=redis_port,
                ttl_hours=48
            )
            logger.info("[SESSION MANAGER] Redis connected")
            
            # Weaviate for long-term memory (Q&A + summary)
            self.weaviate_manager = WeaviateMemoryManager(
                url=weaviate_url,
                api_key=weaviate_api_key
            )
            logger.info("[SESSION MANAGER] Weaviate connected")
            
            # Setup schemas (DocumentMetadata + DocumentQA)
            self._setup_schema()
            
            logger.info("[SESSION MANAGER] Initialization complete")
            
        except Exception as e:
            logger.error(f"[SESSION MANAGER] Initialization failed: {e}")
            raise SessionException(
                "Failed to initialize session manager",
                sys,
                component="initialization"
            )
    
    def _setup_schema(self):
        """Setup Weaviate collections for document management."""
        try:
            client = self.weaviate_manager.client
            if not client:
                logger.error("[SCHEMA] Skipped — no Weaviate connection")
                return

            # ---------------- DocumentMetadata (legacy – keep but unused) ----------------
            if not client.collections.exists("DocumentMetadata"):
                client.collections.create(
                    name="DocumentMetadata",
                    properties=[
                        Property(name="doc_id", data_type=DataType.TEXT),
                        Property(name="filename", data_type=DataType.TEXT),
                        Property(name="filename_hash", data_type=DataType.TEXT),
                        Property(name="file_type", data_type=DataType.TEXT),
                        Property(name="upload_timestamp", data_type=DataType.DATE),
                        Property(name="page_count", data_type=DataType.INT),
                        Property(name="user_id", data_type=DataType.TEXT),
                        Property(name="has_tables", data_type=DataType.BOOL),
                        Property(name="has_charts", data_type=DataType.BOOL),
                    ]
                )
                logger.info("[SCHEMA] DocumentMetadata created")

            # ---------------- DocumentQA (doc-level Q&A) ----------------
            if not client.collections.exists("DocumentQA"):
                client.collections.create(
                    name="DocumentQA",
                    properties=[
                        Property(name="doc_id", data_type=DataType.TEXT),
                        Property(name="user_id", data_type=DataType.TEXT),
                        Property(name="question", data_type=DataType.TEXT),
                        Property(name="answer", data_type=DataType.TEXT),
                        Property(name="timestamp", data_type=DataType.DATE),
                        Property(name="sources", data_type=DataType.TEXT),
                    ]
                )
                logger.info("[SCHEMA] DocumentQA created")

        except Exception as e:
            logger.error(f"[SCHEMA] Setup failed: {e}")
            raise SessionException(
                "Failed to setup Weaviate schema",
                sys,
                component="schema"
            )

    # -------------------------------------------------------------------------
    # LEGACY METADATA HELPERS (Weaviate) – Postgres is now source of truth
    # -------------------------------------------------------------------------
    
    @log_execution_time
    def register_document(
        self,
        filename: str,
        file_type: str,
        user_id: str,
        page_count: int = 0,
        has_tables: bool = False,
        has_charts: bool = False
    ) -> str:
        """
        Register new document in Weaviate (legacy metadata store).
        Not used anymore as Postgres is the source of truth, but kept
        for compatibility.
        """
        try:
            # Validate inputs
            if not filename or not filename.strip():
                raise ValidationException(
                    "Filename cannot be empty",
                    sys,
                    filename=filename
                )
            
            if not user_id or not user_id.strip():
                raise ValidationException(
                    "User ID cannot be empty",
                    sys,
                    user_id=user_id
                )
            
            # Generate unique document ID
            doc_id = f"trade_doc_{self._generate_hash(filename, user_id)}"
            filename_hash = self._hash_filename(filename)
            
            # Create timestamp
            now_utc = datetime.now(timezone.utc)
            timestamp = now_utc.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            
            logger.info(f"[REGISTER] Registering document: {doc_id}")
            
            # Store in Weaviate
            collection = self.weaviate_manager.client.collections.get("DocumentMetadata")
            
            collection.data.insert(
                properties={
                    "doc_id": doc_id,
                    "filename": filename,
                    "filename_hash": filename_hash,
                    "file_type": file_type,
                    "upload_timestamp": timestamp,
                    "page_count": page_count,
                    "user_id": user_id,
                    "has_tables": has_tables,
                    "has_charts": has_charts
                }
            )
            
            logger.info(f"[REGISTER] Document registered: {doc_id} | Pages: {page_count}")
            
            return doc_id
            
        except ValidationException:
            raise
        except Exception as e:
            logger.error(f"[REGISTER] Failed: {e}")
            raise SessionException(
                "Failed to register document",
                sys,
                filename=filename,
                user_id=user_id
            )
    
    @log_execution_time
    def find_document_by_filename(
        self,
        filename: str,
        user_id: str
    ) -> Optional[Dict]:
        """
        Find existing document by filename (duplicate detection).
        Legacy – you now use Postgres hash-based dedupe.
        """
        try:
            filename_hash = self._hash_filename(filename)
            
            logger.info(f"[FIND DOC] Searching for: {filename}")
            
            collection = self.weaviate_manager.client.collections.get("DocumentMetadata")
            
            response = collection.query.fetch_objects(
                filters=Filter.by_property("filename_hash").equal(filename_hash),
                limit=1
            )
            
            if response.objects:
                props = response.objects[0].properties
                logger.info(f"[FIND DOC] Found existing: {props.get('doc_id')}")
                
                return {
                    'doc_id': props.get('doc_id'),
                    'filename': props.get('filename'),
                    'upload_timestamp': props.get('upload_timestamp'),
                    'page_count': props.get('page_count'),
                    'has_tables': props.get('has_tables'),
                    'has_charts': props.get('has_charts')
                }
            
            logger.info(f"[FIND DOC] Not found: {filename}")
            return None
            
        except Exception as e:
            logger.error(f"[FIND DOC] Error: {e}")
            return None
    
    @log_execution_time
    def get_document_metadata(self, doc_id: str) -> Optional[Dict]:
        """
        Get document metadata from Weaviate (legacy).
        Postgres is now the primary metadata store.
        """
        try:
            logger.info(f"[GET METADATA] Doc: {doc_id}")
            
            collection = self.weaviate_manager.client.collections.get("DocumentMetadata")
            
            response = collection.query.fetch_objects(
                filters=Filter.by_property("doc_id").equal(doc_id),
                limit=1
            )
            
            if response.objects:
                props = response.objects[0].properties
                return {
                    'doc_id': props.get('doc_id'),
                    'filename': props.get('filename'),
                    'file_type': props.get('file_type'),
                    'upload_timestamp': props.get('upload_timestamp'),
                    'page_count': props.get('page_count'),
                    'has_tables': props.get('has_tables'),
                    'has_charts': props.get('has_charts')
                }
            
            return None
            
        except Exception as e:
            logger.error(f"[GET METADATA] Error: {e}")
            return None
    
    @log_execution_time
    def get_user_documents(self, user_id: str, limit: int = 50) -> List[Dict]:
        """
        Get all documents for user from Weaviate (legacy).
        """
        try:
            logger.info(f"[GET USER DOCS] User: {user_id} | Limit: {limit}")
            
            collection = self.weaviate_manager.client.collections.get("DocumentMetadata")
            
            response = collection.query.fetch_objects(
                filters=Filter.by_property("user_id").equal(user_id),
                limit=limit
            )
            
            documents = []
            for item in response.objects:
                documents.append({
                    'doc_id': item.properties.get('doc_id'),
                    'filename': item.properties.get('filename'),
                    'file_type': item.properties.get('file_type'),
                    'upload_timestamp': item.properties.get('upload_timestamp'),
                    'page_count': item.properties.get('page_count')
                })
            
            logger.info(f"[GET USER DOCS] Found {len(documents)} documents")
            
            return documents
            
        except Exception as e:
            logger.error(f"[GET USER DOCS] Error: {e}")
            return []

    # -------------------------------------------------------------------------
    # Q&A STORAGE (Weaviate.DocumentQA) – used by main.py + workflow
    # -------------------------------------------------------------------------
    
    @log_execution_time
    def store_document_qa(
        self,
        doc_id: str,
        user_id: str,
        question: str,
        answer: str,
        sources: List[Dict] = None
    ):
        """
        Store Q&A interaction in Weaviate LTM (DocumentQA).
        
        Args:
            doc_id: Document ID
            user_id: User ID
            question: User question
            answer: Generated answer
            sources: Source citations (list of dicts)
        """
        try:
            now_utc = datetime.now(timezone.utc)
            timestamp = now_utc.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            
            logger.info(f"[STORE Q&A] Doc: {doc_id}")
            
            collection = self.weaviate_manager.client.collections.get("DocumentQA")
            
            collection.data.insert(
                properties={
                    "doc_id": doc_id,
                    "user_id": user_id,
                    "question": question,
                    "answer": answer,
                    "timestamp": timestamp,
                    "sources": json.dumps(sources or [])
                }
            )
            
            logger.info("[STORE Q&A] Stored successfully")
            
        except Exception as e:
            logger.error(f"[STORE Q&A] Error: {e}")
            # Don't raise - Q&A storage is non-critical

    @log_execution_time
    def get_last_document_qa(self, doc_id: str, limit: int = 3) -> List[Dict]:
        """
        Return last N Q&A pairs for a document from Weaviate.
        Used by: /get-document-chat-preview
        """
        try:
            logger.info(f"[GET LAST QA] Doc: {doc_id} | Limit: {limit}")
            
            # Use GraphQL-style query API for sorting by timestamp desc
            result = (
                self.weaviate_manager.client.query
                .get("DocumentQA", ["question", "answer", "timestamp"])
                .with_where({
                    "path": ["doc_id"],
                    "operator": "Equal",
                    "valueString": doc_id
                })
                .with_sort([{
                    "path": ["timestamp"],
                    "order": "desc"
                }])
                .with_limit(limit)
                .do()
            )

            items = (
                result.get("data", {})
                      .get("Get", {})
                      .get("DocumentQA", [])
            )

            chats = [
                {
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", ""),
                    "timestamp": qa.get("timestamp", "")
                }
                for qa in items
            ]

            logger.info(f"[GET LAST QA] Retrieved {len(chats)} items")
            return chats

        except Exception as e:
            logger.error(f"[GET LAST QA] Error: {e}")
            return []

    @log_execution_time
    def get_document_qa_count(self, doc_id: str) -> int:
        """
        Count total number of Q&A rows for a document in Weaviate.
        Used by: /get-document-chat-preview, /get-document-summary
        """
        try:
            logger.info(f"[GET QA COUNT] Doc: {doc_id}")
            
            result = (
                self.weaviate_manager.client.query
                .aggregate("DocumentQA")
                .with_where({
                    "path": ["doc_id"],
                    "operator": "Equal",
                    "valueString": doc_id
                })
                .with_fields("meta { count }")
                .do()
            )

            meta = (
                result.get("data", {})
                      .get("Aggregate", {})
                      .get("DocumentQA", [{}])[0]
                      .get("meta", {})
            )
            count = meta.get("count", 0)

            logger.info(f"[GET QA COUNT] Count: {count}")
            return count

        except Exception as e:
            logger.error(f"[GET QA COUNT] Error: {e}")
            return 0

    @log_execution_time
    def get_document_summary(self, doc_id: str, limit: int = 5) -> Optional[str]:
        """
        Get conversation summary for document.
        Currently: lightweight summary built from recent Q&A in DocumentQA.
        
        Args:
            doc_id: Document ID
            limit: Number of recent Q&As to include
        
        Returns:
            Summary string or None
        """
        try:
            logger.info(f"[GET SUMMARY] Doc: {doc_id}")
            
            collection = self.weaviate_manager.client.collections.get("DocumentQA")
            
            response = collection.query.fetch_objects(
                filters=Filter.by_property("doc_id").equal(doc_id),
                limit=limit
            )
            
            if response.objects:
                summary = f"Recent conversations with this document ({len(response.objects)} Q&As):\n\n"
                
                for i, item in enumerate(response.objects, 1):
                    q = item.properties.get('question', '')[:60]
                    timestamp = item.properties.get('timestamp', '')
                    summary += f"{i}. {q}... (at {timestamp[:10]})\n"
                
                logger.info(f"[GET SUMMARY] Generated summary with {len(response.objects)} Q&As")
                
                return summary
            
            return None
            
        except Exception as e:
            logger.error(f"[GET SUMMARY] Error: {e}")
            return None

    # -------------------------------------------------------------------------
    # FULL CLEANUP – used by /delete-document/{doc_id}
    # -------------------------------------------------------------------------

    @log_execution_time
    def delete_document_history(self, doc_id: str):
        """
        Delete all memory for a given document:
        - Redis STM messages
        - Weaviate DocumentQA entries
        - Weaviate ConversationMemory entries (if present)
        - Weaviate DocumentMetadata entries (legacy)
        """
        try:
            logger.info(f"[DELETE HISTORY] Doc: {doc_id}")

            # 1) Clear STM from Redis
            try:
                self.redis_manager.clear_session(doc_id)
                logger.info("[DELETE HISTORY] Redis STM cleared")
            except Exception as e:
                logger.warning(f"[DELETE HISTORY] Failed to clear Redis STM: {e}")

            client = self.weaviate_manager.client
            if not client:
                logger.warning("[DELETE HISTORY] No Weaviate client. Skipping LTM delete.")
                return

            # 2) Delete from DocumentQA
            try:
                if client.collections.exists("DocumentQA"):
                    qa_coll = client.collections.get("DocumentQA")
                    qa_coll.data.delete_many(
                        Filter.by_property("doc_id").equal(doc_id)
                    )
                    logger.info("[DELETE HISTORY] DocumentQA entries deleted")
            except Exception as e:
                logger.warning(f"[DELETE HISTORY] Failed to delete DocumentQA: {e}")

            # 3) Delete from ConversationMemory (summaries), if collection exists
            try:
                if client.collections.exists("ConversationMemory"):
                    conv_coll = client.collections.get("ConversationMemory")
                    conv_coll.data.delete_many(
                        Filter.by_property("doc_id").equal(doc_id)
                    )
                    logger.info("[DELETE HISTORY] ConversationMemory entries deleted")
            except Exception as e:
                logger.warning(f"[DELETE HISTORY] Failed to delete ConversationMemory: {e}")

            # 4) Delete from DocumentMetadata (legacy mirror) if exists
            try:
                if client.collections.exists("DocumentMetadata"):
                    meta_coll = client.collections.get("DocumentMetadata")
                    meta_coll.data.delete_many(
                        Filter.by_property("doc_id").equal(doc_id)
                    )
                    logger.info("[DELETE HISTORY] DocumentMetadata entries deleted")
            except Exception as e:
                logger.warning(f"[DELETE HISTORY] Failed to delete DocumentMetadata: {e}")

        except Exception as e:
            logger.error(f"[DELETE HISTORY] Error: {e}")

    # -------------------------------------------------------------------------
    # UTILITIES
    # -------------------------------------------------------------------------
    
    def _generate_hash(self, filename: str, user_id: str) -> str:
        """
        Generate unique document ID hash
        
        Args:
            filename: Filename
            user_id: User ID
        
        Returns:
            12-character hash
        """
        timestamp = datetime.now().isoformat()
        hash_input = f"{filename}_{user_id}_{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _hash_filename(self, filename: str) -> str:
        """
        Hash filename for duplicate detection
        """
        return hashlib.md5(filename.encode()).hexdigest()
    
    def close(self):
        """Close connections"""
        try:
            logger.info("[SESSION MANAGER] Closing connections...")
            if self.weaviate_manager:
                self.weaviate_manager.close()
            logger.info("[SESSION MANAGER] Connections closed")
        except Exception as e:
            logger.error(f"[SESSION MANAGER] Error closing: {e}")


# Export
__all__ = ['DocumentSessionManager']
