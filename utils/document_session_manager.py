"""
Document Session Manager
Path: utils/document_session_manager.py

Handles document-specific chat sessions with Redis (STM) and Weaviate (LTM)
"""

import json
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone, timedelta
import hashlib
from pathlib import Path

# Import from your existing memory_manager
from utils.memory_manager import RedisMemoryManager, WeaviateMemoryManager
from custom_logging.my_logger import logger
from exception.exceptions import TradingBotException
import sys

from weaviate.classes.config import Property, DataType
from weaviate.classes.query import Filter


class DocumentMetadata:
    """Document metadata structure"""
    
    def __init__(
        self,
        doc_id: str,
        filename: str,
        file_type: str,
        upload_timestamp: str,
        page_count: int = 0,
        is_ocr: bool = False,
        processing_status: str = "pending"
    ):
        self.doc_id = doc_id
        self.filename = filename
        self.file_type = file_type
        self.upload_timestamp = upload_timestamp
        self.page_count = page_count
        self.is_ocr = is_ocr
        self.processing_status = processing_status
    
    def to_dict(self) -> Dict:
        return {
            'doc_id': self.doc_id,
            'filename': self.filename,
            'file_type': self.file_type,
            'upload_timestamp': self.upload_timestamp,
            'page_count': self.page_count,
            'is_ocr': self.is_ocr,
            'processing_status': self.processing_status
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


class DocumentSessionManager:
    """
    Manages document-specific chat sessions
    - Redis: Active document sessions and recent Q&A
    - Weaviate: Long-term document analysis history
    """
    
    def __init__(
        self,
        redis_host: str = 'localhost',
        redis_port: int = 6380,
        weaviate_url: str = 'http://localhost:8080',
        weaviate_api_key: Optional[str] = None,
        session_ttl_hours: int = 48
    ):
        logger.info("Initializing Document Session Manager...")
        
        # Initialize Redis for short-term memory
        self.redis_manager = RedisMemoryManager(
            host=redis_host,
            port=redis_port,
            ttl_hours=session_ttl_hours
        )
        
        # Initialize Weaviate for long-term memory
        self.weaviate_manager = WeaviateMemoryManager(
            url=weaviate_url,
            api_key=weaviate_api_key
        )
        
        # Setup document-specific schema
        self._setup_document_schema()
        
        logger.info("Document Session Manager initialized")
    
    def _setup_document_schema(self):
        """Setup Weaviate schema for document sessions"""
        try:
            # Document metadata collection
            if not self.weaviate_manager.client.collections.exists("DocumentMetadata"):
                self.weaviate_manager.client.collections.create(
                    name="DocumentMetadata",
                    properties=[
                        Property(name="doc_id", data_type=DataType.TEXT, description="Unique document ID"),
                        Property(name="filename", data_type=DataType.TEXT, description="Original filename"),
                        Property(name="file_type", data_type=DataType.TEXT, description="File type (pdf, docx, image)"),
                        Property(name="upload_timestamp", data_type=DataType.DATE, description="Upload time"),
                        Property(name="page_count", data_type=DataType.INT, description="Number of pages"),
                        Property(name="is_ocr", data_type=DataType.BOOL, description="Was OCR used"),
                        Property(name="processing_status", data_type=DataType.TEXT, description="Processing status"),
                        Property(name="user_id", data_type=DataType.TEXT, description="Owner user ID"),
                        Property(name="tags", data_type=DataType.TEXT_ARRAY, description="Document tags"),
                    ]
                )
            
            # Document Q&A history collection
            if not self.weaviate_manager.client.collections.exists("DocumentQA"):
                self.weaviate_manager.client.collections.create(
                    name="DocumentQA",
                    properties=[
                        Property(name="doc_id", data_type=DataType.TEXT, description="Associated document ID"),
                        Property(name="session_id", data_type=DataType.TEXT, description="Session ID"),
                        Property(name="user_id", data_type=DataType.TEXT, description="User ID"),
                        Property(name="question", data_type=DataType.TEXT, description="User question"),
                        Property(name="answer", data_type=DataType.TEXT, description="Bot answer"),
                        Property(name="timestamp", data_type=DataType.DATE, description="Q&A timestamp"),
                        Property(name="relevance_score", data_type=DataType.NUMBER, description="Answer relevance"),
                        Property(name="sources", data_type=DataType.TEXT, description="Source pages/sections as JSON"),
                    ]
                )
            
            logger.info("Document schema setup complete")
            
        except Exception as e:
            logger.error(f"Document schema setup failed: {e}")
            raise TradingBotException(e, sys)
    
    # ==================== Document Management ====================
    
    def register_document(
        self,
        filename: str,
        file_type: str,
        user_id: str,
        page_count: int = 0,
        is_ocr: bool = False,
        tags: List[str] = None
    ) -> str:
        """
        Register a new document in the system
        
        Returns:
            doc_id: Unique document identifier
        """
        try:
            # Generate unique doc_id
            doc_id = self._generate_doc_id(filename, user_id)
            
            now_utc = datetime.now(timezone.utc)
            timestamp_string = now_utc.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            
            # Store in Weaviate
            collection = self.weaviate_manager.client.collections.get("DocumentMetadata")
            
            collection.data.insert(
                properties={
                    "doc_id": doc_id,
                    "filename": filename,
                    "file_type": file_type,
                    "upload_timestamp": timestamp_string,
                    "page_count": page_count,
                    "is_ocr": is_ocr,
                    "processing_status": "completed",
                    "user_id": user_id,
                    "tags": tags or []
                }
            )
            
            # Store in Redis for quick access
            redis_key = f"document:{doc_id}"
            doc_data = {
                'doc_id': doc_id,
                'filename': filename,
                'file_type': file_type,
                'upload_timestamp': timestamp_string,
                'page_count': page_count,
                'is_ocr': is_ocr,
                'user_id': user_id
            }
            self.redis_manager.redis_client.set(
                redis_key,
                json.dumps(doc_data),
                ex=172800  # 48 hours
            )
            
            logger.info(f"Registered document: {doc_id} - {filename}")
            
            return doc_id
            
        except Exception as e:
            logger.error(f"Document registration failed: {e}")
            raise TradingBotException(e, sys)
    
    def get_user_documents(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get all documents for a user"""
        try:
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
                    'page_count': item.properties.get('page_count'),
                    'is_ocr': item.properties.get('is_ocr'),
                    'processing_status': item.properties.get('processing_status')
                })
            
            logger.info(f"Retrieved {len(documents)} documents for user {user_id}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to get user documents: {e}")
            return []
    
    def get_document_metadata(self, doc_id: str) -> Optional[Dict]:
        """Get metadata for a specific document"""
        try:
            # Try Redis first (faster)
            redis_key = f"document:{doc_id}"
            cached_data = self.redis_manager.redis_client.get(redis_key)
            
            if cached_data:
                return json.loads(cached_data)
            
            # Fallback to Weaviate
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
                    'is_ocr': props.get('is_ocr')
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document metadata: {e}")
            return None
    
    # ==================== Session Management ====================
    
    def create_document_session(
        self,
        doc_id: str,
        user_id: str
    ) -> str:
        """
        Create a new document chat session
        
        Returns:
            session_id: Unique session identifier
        """
        try:
            session_id = f"doc_{doc_id}_{user_id}_{int(datetime.now().timestamp())}"
            
            # Initialize session in Redis
            session_key = f"doc_session:{session_id}"
            session_data = {
                'session_id': session_id,
                'doc_id': doc_id,
                'user_id': user_id,
                'created_at': datetime.now().isoformat(),
                'message_count': 0,
                'active_document': doc_id
            }
            
            self.redis_manager.redis_client.set(
                session_key,
                json.dumps(session_data),
                ex=172800  # 48 hours
            )
            
            logger.info(f"Created document session: {session_id}")
            
            return session_id
            
        except Exception as e:
            logger.error(f"Session creation failed: {e}")
            raise TradingBotException(e, sys)
    
    def store_document_qa(
        self,
        session_id: str,
        doc_id: str,
        user_id: str,
        question: str,
        answer: str,
        relevance_score: float = 0.0,
        sources: List[Dict] = None
    ):
        """Store Q&A interaction for a document"""
        try:
            # Store in Redis (short-term)
            qa_key = f"doc_qa:{session_id}"
            qa_entry = {
                'question': question,
                'answer': answer,
                'timestamp': datetime.now().isoformat(),
                'relevance_score': relevance_score
            }
            
            self.redis_manager.redis_client.rpush(qa_key, json.dumps(qa_entry))
            self.redis_manager.redis_client.expire(qa_key, 172800)
            
            # Store in Weaviate (long-term)
            now_utc = datetime.now(timezone.utc)
            timestamp_string = now_utc.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            
            collection = self.weaviate_manager.client.collections.get("DocumentQA")
            
            collection.data.insert(
                properties={
                    "doc_id": doc_id,
                    "session_id": session_id,
                    "user_id": user_id,
                    "question": question,
                    "answer": answer,
                    "timestamp": timestamp_string,
                    "relevance_score": relevance_score,
                    "sources": json.dumps(sources or [])
                }
            )
            
            logger.info(f"Stored Q&A for document {doc_id}")
            
        except Exception as e:
            logger.error(f"Failed to store document Q&A: {e}")
    
    def get_document_qa_history(
        self,
        doc_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """Get Q&A history for a document"""
        try:
            collection = self.weaviate_manager.client.collections.get("DocumentQA")
            
            response = collection.query.fetch_objects(
                filters=Filter.by_property("doc_id").equal(doc_id),
                limit=limit
            )
            
            qa_history = []
            for item in response.objects:
                qa_history.append({
                    'question': item.properties.get('question'),
                    'answer': item.properties.get('answer'),
                    'timestamp': item.properties.get('timestamp'),
                    'relevance_score': item.properties.get('relevance_score')
                })
            
            return qa_history
            
        except Exception as e:
            logger.error(f"Failed to get Q&A history: {e}")
            return []
    
    def set_active_document(self, session_id: str, doc_id: str):
        """Set the active document for a session"""
        try:
            session_key = f"active_doc:{session_id}"
            self.redis_manager.redis_client.set(
                session_key,
                doc_id,
                ex=172800
            )
            logger.info(f"Set active document: {doc_id} for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to set active document: {e}")
    
    def get_active_document(self, session_id: str) -> Optional[str]:
        """Get the currently active document for a session"""
        try:
            session_key = f"active_doc:{session_id}"
            doc_id = self.redis_manager.redis_client.get(session_key)
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to get active document: {e}")
            return None
    
    # ==================== Utility Methods ====================
    
    def _generate_doc_id(self, filename: str, user_id: str) -> str:
        """Generate unique document ID"""
        timestamp = datetime.now().isoformat()
        hash_input = f"{filename}_{user_id}_{timestamp}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def search_document_qa(
        self,
        query: str,
        doc_id: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict]:
        """Search through document Q&A history"""
        try:
            collection = self.weaviate_manager.client.collections.get("DocumentQA")
            
            # Build filter
            filters = None
            if doc_id:
                filters = Filter.by_property("doc_id").equal(doc_id)
            
            response = collection.query.near_text(
                query=query,
                limit=limit,
                filters=filters
            )
            
            results = []
            for item in response.objects:
                results.append({
                    'doc_id': item.properties.get('doc_id'),
                    'question': item.properties.get('question'),
                    'answer': item.properties.get('answer'),
                    'relevance': item.metadata.distance
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Document Q&A search failed: {e}")
            return []
    
    def close(self):
        """Close connections"""
        try:
            self.weaviate_manager.close()
            logger.info("Document Session Manager closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")


# Export
__all__ = ['DocumentSessionManager', 'DocumentMetadata']