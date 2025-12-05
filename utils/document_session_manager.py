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
    - Document metadata (Weaviate LTM)
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
            
            # Redis for short-term memory
            self.redis_manager = RedisMemoryManager_document(
                host=redis_host,
                port=redis_port,
                ttl_hours=48
            )
            logger.info("[SESSION MANAGER] Redis connected")
            
            # Weaviate for long-term memory
            self.weaviate_manager = WeaviateMemoryManager(
                url=weaviate_url,
                api_key=weaviate_api_key
            )
            logger.info("[SESSION MANAGER] Weaviate connected")
            
            # Setup schemas
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
        """Setup Weaviate collections for document management"""
        try:
            client = self.weaviate_manager.client
            
            # Collection 1: DocumentMetadata
            if not client.collections.exists("DocumentMetadata"):
                client.collections.create(
                    name="DocumentMetadata",
                    properties=[
                        Property(name="doc_id", data_type=DataType.TEXT, description="Unique document ID"),
                        Property(name="filename", data_type=DataType.TEXT, description="Original filename"),
                        Property(name="filename_hash", data_type=DataType.TEXT, description="Hash for duplicate detection"),
                        Property(name="file_type", data_type=DataType.TEXT, description="File type (pdf, docx, etc)"),
                        Property(name="upload_timestamp", data_type=DataType.DATE, description="Upload time"),
                        Property(name="page_count", data_type=DataType.INT, description="Number of pages"),
                        Property(name="user_id", data_type=DataType.TEXT, description="User ID"),
                        Property(name="has_tables", data_type=DataType.BOOL, description="Contains tables"),
                        Property(name="has_charts", data_type=DataType.BOOL, description="Contains charts"),
                    ]
                )
                logger.info("[SCHEMA] DocumentMetadata collection created")
            
            # Collection 2: DocumentQA
            if not client.collections.exists("DocumentQA"):
                client.collections.create(
                    name="DocumentQA",
                    properties=[
                        Property(name="doc_id", data_type=DataType.TEXT, description="Document ID"),
                        Property(name="session_id", data_type=DataType.TEXT, description="Session ID"),
                        Property(name="user_id", data_type=DataType.TEXT, description="User ID"),
                        Property(name="question", data_type=DataType.TEXT, description="User question"),
                        Property(name="answer", data_type=DataType.TEXT, description="Generated answer"),
                        Property(name="timestamp", data_type=DataType.DATE, description="Q&A timestamp"),
                        Property(name="sources", data_type=DataType.TEXT, description="Source citations (JSON)"),
                    ]
                )
                logger.info("[SCHEMA] DocumentQA collection created")
            
            logger.info("[SCHEMA] All collections verified/created")
            
        except Exception as e:
            logger.error(f"[SCHEMA] Setup failed: {e}")
            raise SessionException(
                "Failed to setup Weaviate schema",
                sys,
                component="schema"
            )
    
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
        Register new document in Weaviate
        
        Args:
            filename: Original filename
            file_type: File extension (pdf, docx, etc)
            user_id: User ID
            page_count: Number of pages
            has_tables: Whether document has tables
            has_charts: Whether document has charts
        
        Returns:
            Generated document ID (format: trade_doc_{hash})
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
        Find existing document by filename (duplicate detection)
        
        Args:
            filename: Original filename
            user_id: User ID
        
        Returns:
            Document metadata dict or None
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
        Get document metadata
        
        Args:
            doc_id: Document ID
        
        Returns:
            Document metadata or None
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
        Get all documents for user
        
        Args:
            user_id: User ID
            limit: Maximum documents to return
        
        Returns:
            List of document metadata
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
    
    @log_execution_time
    def store_document_qa(
        self,
        session_id: str,
        doc_id: str,
        user_id: str,
        question: str,
        answer: str,
        sources: List[Dict] = None
    ):
        """
        Store Q&A interaction in Weaviate LTM
        
        Args:
            session_id: Session ID
            doc_id: Document ID
            user_id: User ID
            question: User question
            answer: Generated answer
            sources: Source citations
        """
        try:
            now_utc = datetime.now(timezone.utc)
            timestamp = now_utc.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            
            logger.info(f"[STORE Q&A] Doc: {doc_id} | Session: {session_id}")
            
            collection = self.weaviate_manager.client.collections.get("DocumentQA")
            
            collection.data.insert(
                properties={
                    "doc_id": doc_id,
                    "session_id": session_id,
                    "user_id": user_id,
                    "question": question,
                    "answer": answer,
                    "timestamp": timestamp,
                    "sources": json.dumps(sources or [])
                }
            )
            
            logger.info(f"[STORE Q&A] Stored successfully")
            
        except Exception as e:
            logger.error(f"[STORE Q&A] Error: {e}")
            # Don't raise - Q&A storage is non-critical
    
    @log_execution_time
    def get_document_summary(self, doc_id: str, limit: int = 5) -> Optional[str]:
        """
        Get conversation summary for document
        
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
    
    @log_execution_time
    def get_document_qa_count(self, doc_id: str) -> int:
        """
        Get Q&A count for document
        
        Args:
            doc_id: Document ID
        
        Returns:
            Number of Q&As
        """
        try:
            collection = self.weaviate_manager.client.collections.get("DocumentQA")
            
            response = collection.query.fetch_objects(
                filters=Filter.by_property("doc_id").equal(doc_id),
                limit=100
            )
            
            count = len(response.objects)
            logger.info(f"[GET QA COUNT] Doc: {doc_id} | Count: {count}")
            
            return count
            
        except Exception as e:
            logger.error(f"[GET QA COUNT] Error: {e}")
            return 0
    
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
        
        Args:
            filename: Filename
        
        Returns:
            MD5 hash
        """
        return hashlib.md5(filename.encode()).hexdigest()
    
    def close(self):
        """Close connections"""
        try:
            logger.info("[SESSION MANAGER] Closing connections...")
            self.weaviate_manager.close()
            logger.info("[SESSION MANAGER] Connections closed")
        except Exception as e:
            logger.error(f"[SESSION MANAGER] Error closing: {e}")


# Export
__all__ = ['DocumentSessionManager']