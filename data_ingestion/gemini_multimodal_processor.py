"""
Gemini Multimodal Document Processor
- Uses Gemini Flash for complete document processing
- Extracts text, tables, and charts
- Stores in Pinecone with proper metadata
- Handles PDFs, DOCX, and images
"""

import os
import sys
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import base64
from io import BytesIO

# Document processing
import pymupdf  # PyMuPDF
from PIL import Image
from pdf2image import convert_from_path
import docx

# Google Gemini
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Vector store
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from uuid import uuid4

# Internal imports
from custom_logging.my_logger import logger, log_execution_time
from exception.exceptions import (
    DocumentProcessingException,
    VectorStoreException,
    TradingBotException
)


class GeminiMultimodalProcessor:
    """
    Complete document processor using Gemini Flash
    - Text extraction with OCR fallback
    - Table detection and parsing
    - Chart/diagram interpretation
    - Multimodal analysis
    """
    
    def __init__(self):
        """Initialize Gemini processor"""
        try:
            logger.info("[GEMINI PROCESSOR] Initializing...")
            
            # Configure Gemini
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment")
            
            genai.configure(api_key=api_key)
            
            # Initialize Gemini Flash model
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("[GEMINI] Model initialized: gemini-1.5-flash")
            
            # Safety settings (allow all for document processing)
            self.safety_settings = {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
            
            # Embeddings for Pinecone
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004"
            )
            logger.info("[GEMINI] Embeddings initialized")
            
            # Pinecone setup
            self._setup_pinecone()
            
            logger.info("[GEMINI PROCESSOR] Initialization complete")
            
        except Exception as e:
            logger.error(f"[GEMINI PROCESSOR] Initialization failed: {e}")
            raise TradingBotException(
                "Failed to initialize Gemini processor",
                sys,
                component="gemini_processor"
            )
    
    def _setup_pinecone(self):
        """Setup Pinecone connection"""
        try:
            api_key = os.getenv('PINECONE_API_KEY')
            if not api_key:
                raise ValueError("PINECONE_API_KEY not found")
            
            self.pc = Pinecone(api_key=api_key)
            self.index = self.pc.Index("trading-bot")
            
            logger.info("[PINECONE] Connected successfully")
            
        except Exception as e:
            logger.error(f"[PINECONE] Setup failed: {e}")
            raise VectorStoreException(
                "Failed to setup Pinecone",
                sys,
                service="pinecone"
            )
    
    @log_execution_time
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process document with Gemini multimodal analysis
        
        Args:
            file_path: Path to document file
        
        Returns:
            Dict containing:
            - text_chunks: List of text content
            - tables: List of detected tables
            - charts: List of detected charts/diagrams
            - page_count: Number of pages
        """
        try:
            file_ext = Path(file_path).suffix.lower()
            logger.info(f"[GEMINI] Processing document: {file_path}")
            
            if file_ext == '.pdf':
                return self._process_pdf(file_path)
            elif file_ext == '.docx':
                return self._process_docx(file_path)
            else:
                raise DocumentProcessingException(
                    f"Unsupported file type: {file_ext}",
                    sys,
                    file_path=file_path
                )
                
        except DocumentProcessingException:
            raise
        except Exception as e:
            logger.error(f"[GEMINI] Document processing failed: {e}")
            raise DocumentProcessingException(
                "Failed to process document",
                sys,
                file_path=file_path
            )
    
    @log_execution_time
    def _process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process PDF with Gemini"""
        try:
            logger.info(f"[PDF] Processing: {pdf_path}")
            
            # Convert PDF to images for Gemini
            images = convert_from_path(pdf_path, dpi=150)
            logger.info(f"[PDF] Converted to {len(images)} images")
            
            # Open PDF for metadata
            pdf_document = pymupdf.open(pdf_path)
            page_count = len(pdf_document)
            
            text_chunks = []
            tables = []
            charts = []
            
            # Process each page
            for page_num, image in enumerate(images, start=1):
                logger.info(f"[PDF] Processing page {page_num}/{page_count}")
                
                # Convert PIL Image to bytes
                img_byte_arr = BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Analyze page with Gemini
                page_analysis = self._analyze_page_with_gemini(
                    img_byte_arr,
                    page_num
                )
                
                # Categorize content
                if page_analysis['text']:
                    text_chunks.append({
                        'page_num': page_num,
                        'content': page_analysis['text'],
                        'type': 'text'
                    })
                
                tables.extend(page_analysis['tables'])
                charts.extend(page_analysis['charts'])
            
            pdf_document.close()
            
            logger.info(
                f"[PDF] Processing complete | "
                f"Text chunks: {len(text_chunks)} | "
                f"Tables: {len(tables)} | "
                f"Charts: {len(charts)}"
            )
            
            return {
                'text_chunks': text_chunks,
                'tables': tables,
                'charts': charts,
                'page_count': page_count
            }
            
        except Exception as e:
            logger.error(f"[PDF] Processing failed: {e}")
            raise DocumentProcessingException(
                "Failed to process PDF",
                sys,
                file_path=pdf_path
            )
    
    @log_execution_time
    def _analyze_page_with_gemini(
        self,
        image_bytes: bytes,
        page_num: int
    ) -> Dict[str, Any]:
        """
        Analyze single page with Gemini Flash
        """
        try:
            # Prepare image for Gemini
            image = Image.open(BytesIO(image_bytes))
            
            # Comprehensive analysis prompt
            prompt = """Analyze this document page and provide a structured response:

1. **TEXT CONTENT**: Extract all readable text from the page. Maintain paragraph structure and formatting.

2. **TABLES**: If there are any tables:
   - Identify each table with a unique ID (T1, T2, etc.)
   - Extract the table structure (headers and rows)
   - Describe what the table represents

3. **CHARTS/DIAGRAMS**: If there are any charts, graphs, or diagrams:
   - Identify each with a unique ID (C1, C2, etc.)
   - Describe the type (bar chart, line graph, pie chart, etc.)
   - Extract key data points and trends
   - Explain what the chart represents

**RESPONSE FORMAT:**
Return your analysis in this JSON structure:
```json
{
  "text": "Full text content here...",
  "tables": [
    {
      "table_id": "T1",
      "content": "Table headers and rows...",
      "description": "What the table shows..."
    }
  ],
  "charts": [
    {
      "chart_id": "C1",
      "type": "bar chart",
      "description": "Detailed description with data points...",
      "key_insights": "Main findings..."
    }
  ]
}
```

If there are no tables or charts, return empty arrays for those fields.
Provide ONLY the JSON response, no other text."""
            
            # Call Gemini
            response = self.model.generate_content(
                [prompt, image],
                safety_settings=self.safety_settings
            )
            
            # Parse response
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Parse JSON
            try:
                analysis = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.warning(f"[GEMINI] JSON parse failed, using fallback: {e}")
                # Fallback: extract text directly
                analysis = {
                    'text': response_text,
                    'tables': [],
                    'charts': []
                }
            
            # Add page numbers to tables and charts
            for table in analysis.get('tables', []):
                table['page_num'] = page_num
            
            for chart in analysis.get('charts', []):
                chart['page_num'] = page_num
            
            logger.debug(
                f"[GEMINI] Page {page_num} analyzed | "
                f"Tables: {len(analysis.get('tables', []))} | "
                f"Charts: {len(analysis.get('charts', []))}"
            )
            
            return {
                'text': analysis.get('text', ''),
                'tables': analysis.get('tables', []),
                'charts': analysis.get('charts', [])
            }
            
        except Exception as e:
            logger.error(f"[GEMINI] Page analysis failed: {e}")
            # Return empty analysis on failure
            return {
                'text': '',
                'tables': [],
                'charts': []
            }
    
    @log_execution_time
    def _process_docx(self, docx_path: str) -> Dict[str, Any]:
        """Process DOCX file"""
        try:
            logger.info(f"[DOCX] Processing: {docx_path}")
            
            doc = docx.Document(docx_path)
            
            text_chunks = []
            tables = []
            
            # Extract text
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text)
            
            if full_text:
                text_chunks.append({
                    'page_num': 1,
                    'content': '\n'.join(full_text),
                    'type': 'text'
                })
            
            # Extract tables
            for table_idx, table in enumerate(doc.tables, start=1):
                table_text = []
                for row in table.rows:
                    row_text = [cell.text for cell in row.cells]
                    table_text.append(' | '.join(row_text))
                
                tables.append({
                    'table_id': f'T{table_idx}',
                    'page_num': 1,
                    'content': '\n'.join(table_text),
                    'description': f'Table {table_idx} from DOCX'
                })
            
            logger.info(
                f"[DOCX] Processing complete | "
                f"Text chunks: {len(text_chunks)} | "
                f"Tables: {len(tables)}"
            )
            
            return {
                'text_chunks': text_chunks,
                'tables': tables,
                'charts': [],
                'page_count': 1
            }
            
        except Exception as e:
            logger.error(f"[DOCX] Processing failed: {e}")
            raise DocumentProcessingException(
                "Failed to process DOCX",
                sys,
                file_path=docx_path
            )
    
    @log_execution_time
    def store_in_pinecone(
        self,
        processed_data: Dict[str, Any],
        doc_id: str,
        filename: str
    ):
        """
        Store processed document in Pinecone
        
        Args:
            processed_data: Output from process_document()
            doc_id: Unique document ID
            filename: Original filename
        """
        try:
            logger.info(f"[PINECONE] Storing document: {doc_id}")
            
            vector_store = PineconeVectorStore(
                index=self.index,
                embedding=self.embeddings,
                namespace="documents"
            )
            
            documents = []
            
            # Store text chunks
            for chunk in processed_data.get('text_chunks', []):
                doc = Document(
                    page_content=chunk['content'],
                    metadata={
                        'doc_id': doc_id,
                        'filename': filename,
                        'type': 'text',
                        'page_num': chunk['page_num']
                    }
                )
                documents.append(doc)
            
            # Store tables
            for table in processed_data.get('tables', []):
                doc = Document(
                    page_content=f"{table.get('description', '')}\n\n{table['content']}",
                    metadata={
                        'doc_id': doc_id,
                        'filename': filename,
                        'type': 'table',
                        'page_num': table['page_num'],
                        'table_id': table['table_id'],
                        'table_name': table.get('description', f"Table {table['table_id']}")
                    }
                )
                documents.append(doc)
            
            # Store charts
            for chart in processed_data.get('charts', []):
                doc = Document(
                    page_content=f"{chart.get('description', '')}\n\nKey insights: {chart.get('key_insights', '')}",
                    metadata={
                        'doc_id': doc_id,
                        'filename': filename,
                        'type': 'chart',
                        'page_num': chart['page_num'],
                        'chart_id': chart['chart_id'],
                        'chart_name': chart.get('type', f"Chart {chart['chart_id']}")
                    }
                )
                documents.append(doc)
            
            # Generate UUIDs
            uuids = [str(uuid4()) for _ in range(len(documents))]
            
            # Store in Pinecone
            vector_store.add_documents(documents=documents, ids=uuids)
            
            logger.info(
                f"[PINECONE] Stored {len(documents)} vectors | "
                f"Text: {len(processed_data.get('text_chunks', []))} | "
                f"Tables: {len(processed_data.get('tables', []))} | "
                f"Charts: {len(processed_data.get('charts', []))}"
            )
            
        except Exception as e:
            logger.error(f"[PINECONE] Storage failed: {e}")
            raise VectorStoreException(
                "Failed to store in Pinecone",
                sys,
                doc_id=doc_id
            )


# Export
__all__ = ['GeminiMultimodalProcessor']
