import os
import tempfile
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from utils.model_loaders import ModelLoader
from utils.config_loader import load_config
from pinecone import ServerlessSpec, Pinecone
from uuid import uuid4
import sys
from exception.exceptions import TradingBotException
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
import numpy as np
import pymupdf  # PyMuPDF for better PDF handling
import io
import base64
import json


class MultiModalProcessor:
    """
    Handles multi-modal content processing:
    - Text extraction from PDFs
    - Image extraction and captioning (BLIP)
    - Image embedding generation (CLIP)
    - Table extraction from PDFs
    """
    
    def __init__(self):
        print("Initializing Multi-Modal Processor...")
        
        # Load CLIP for image embeddings (open-source)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Load BLIP for image captioning (open-source)
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model.to(self.device)
        self.blip_model.to(self.device)
        
        print(f"Multi-Modal Processor initialized on {self.device}")
    
    def generate_image_caption(self, image: Image.Image) -> str:
        """Generate caption for an image using BLIP"""
        try:
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            output = self.blip_model.generate(**inputs, max_length=100)
            caption = self.blip_processor.decode(output[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            print(f"Error generating caption: {e}")
            return "Image content"
    
    def generate_image_embedding(self, image: Image.Image) -> List[float]:
        """Generate CLIP embedding for an image"""
        try:
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                # Normalize the embedding
                embedding = image_features / image_features.norm(dim=-1, keepdim=True)
                return embedding.cpu().numpy().flatten().tolist()
        except Exception as e:
            print(f"Error generating image embedding: {e}")
            return None
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract images from PDF with metadata"""
        images_data = []
        
        try:
            pdf_document = pymupdf.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert to PIL Image
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Convert to RGB if needed
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Generate caption and embedding
                    caption = self.generate_image_caption(image)
                    embedding = self.generate_image_embedding(image)
                    
                    # Encode image as base64 for storage
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    images_data.append({
                        'page_num': page_num + 1,
                        'image_index': img_index,
                        'caption': caption,
                        'embedding': embedding,
                        'image_base64': img_str,
                        'image_format': base_image["ext"]
                    })
            
            pdf_document.close()
            print(f"Extracted {len(images_data)} images from PDF")
            
        except Exception as e:
            print(f"Error extracting images from PDF: {e}")
        
        return images_data
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables from PDF"""
        tables_data = []
        
        try:
            import camelot  # Better table extraction
            
            # Extract tables using camelot
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            
            for i, table in enumerate(tables):
                df = table.df
                
                # Convert table to text representation
                table_text = df.to_string(index=False)
                table_markdown = df.to_markdown(index=False)
                
                tables_data.append({
                    'table_index': i,
                    'page': table.page,
                    'text_representation': table_text,
                    'markdown_representation': table_markdown,
                    'shape': df.shape
                })
            
            print(f"Extracted {len(tables_data)} tables from PDF")
            
        except ImportError:
            print("Camelot not installed. Using basic table extraction...")
            # Fallback to basic extraction
            pdf_document = pymupdf.open(pdf_path)
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                tables = page.find_tables()
                
                for i, table in enumerate(tables):
                    if table:
                        table_data = table.extract()
                        table_text = "\n".join(["\t".join(str(cell) for cell in row) for row in table_data])
                        
                        tables_data.append({
                            'table_index': i,
                            'page': page_num + 1,
                            'text_representation': table_text,
                            'markdown_representation': table_text,
                            'shape': (len(table_data), len(table_data[0]) if table_data else 0)
                        })
            
            pdf_document.close()
            
        except Exception as e:
            print(f"Error extracting tables: {e}")
        
        return tables_data


class MultiModalDataIngestion:
    """
    Enhanced data ingestion class with multi-modal support:
    - Text documents (PDF, DOCX)
    - Images (extracted from PDFs or standalone)
    - Tables (extracted from PDFs)
    """

    def __init__(self):
        try:
            print("Initializing Multi-Modal DataIngestion pipeline...")
            self.model_loader = ModelLoader()
            self._load_env_variables()
            self.config = load_config()
            self.multimodal_processor = MultiModalProcessor()
        except Exception as e:
            raise TradingBotException(e, sys)

    def _load_env_variables(self):
        try:
            load_dotenv()

            required_vars = [
                "GOOGLE_API_KEY",
                "PINECONE_API_KEY"
            ]

            missing_vars = [var for var in required_vars if os.getenv(var) is None]
            if missing_vars:
                raise EnvironmentError(f"Missing environment variables: {missing_vars}")

            self.google_api_key = os.getenv("GOOGLE_API_KEY")
            self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        except Exception as e:
            raise TradingBotException(e, sys)

    def load_documents(self, uploaded_files) -> Dict[str, List]:
        """
        Load and process documents with multi-modal support
        
        Returns:
            Dictionary containing:
            - 'text_documents': List of text Document objects
            - 'images': List of image data with embeddings
            - 'tables': List of table data
        """
        try:
            text_documents = []
            all_images = []
            all_tables = []
            
            for uploaded_file in uploaded_files:
                file_ext = os.path.splitext(uploaded_file.filename)[1].lower()
                suffix = file_ext if file_ext in [".pdf", ".docx"] else ".tmp"

                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    temp_file.write(uploaded_file.file.read())
                    temp_path = temp_file.name

                if file_ext == ".pdf":
                    # 1. Load text content
                    loader = PyPDFLoader(temp_path)
                    text_docs = loader.load()
                    text_documents.extend(text_docs)
                    
                    # 2. Extract images
                    images = self.multimodal_processor.extract_images_from_pdf(temp_path)
                    all_images.extend(images)
                    
                    # 3. Extract tables
                    tables = self.multimodal_processor.extract_tables_from_pdf(temp_path)
                    all_tables.extend(tables)
                    
                elif file_ext == ".docx":
                    loader = Docx2txtLoader(temp_path)
                    text_documents.extend(loader.load())
                    
                else:
                    print(f"Unsupported file type: {uploaded_file.filename}")
                
                # Clean up temp file
                os.unlink(temp_path)
            
            print(f"Loaded {len(text_documents)} text documents, {len(all_images)} images, {len(all_tables)} tables")
            
            return {
                'text_documents': text_documents,
                'images': all_images,
                'tables': all_tables
            }
            
        except Exception as e:
            raise TradingBotException(e, sys)

    def store_in_vector_db(self, processed_data: Dict[str, List]):
        """
        Store multi-modal data in Pinecone with proper indexing
        
        Creates three separate namespaces:
        - 'text': For text documents
        - 'images': For image captions and embeddings
        - 'tables': For table content
        """
        try:
            pinecone_client = Pinecone(api_key=self.pinecone_api_key)
            index_name = self.config["vector_db"]["index_name"]

            # Check if index exists, create if not
            if index_name not in [i.name for i in pinecone_client.list_indexes()]:
                pinecone_client.create_index(
                    name=index_name,
                    dimension=768,  # For text embeddings
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
            
            index = pinecone_client.Index(index_name)
            text_embeddings = self.model_loader.load_embeddings()
            
            # 1. Store text documents
            text_documents = processed_data.get('text_documents', [])
            if text_documents:
                print(f"Storing {len(text_documents)} text documents...")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                split_docs = text_splitter.split_documents(text_documents)
                
                vector_store = PineconeVectorStore(
                    index=index, 
                    embedding=text_embeddings,
                    namespace="text"
                )
                uuids = [str(uuid4()) for _ in range(len(split_docs))]
                vector_store.add_documents(documents=split_docs, ids=uuids)
            
            # 2. Store images (captions as text, CLIP embeddings)
            images = processed_data.get('images', [])
            if images:
                print(f"Storing {len(images)} images...")
                
                # Create documents from image captions
                image_docs = []
                for img_data in images:
                    doc = Document(
                        page_content=f"Image: {img_data['caption']}",
                        metadata={
                            'type': 'image',
                            'page_num': img_data['page_num'],
                            'image_index': img_data['image_index'],
                            'image_base64': img_data['image_base64'][:100],  # Store reference only
                            'caption': img_data['caption']
                        }
                    )
                    image_docs.append(doc)
                
                vector_store_images = PineconeVectorStore(
                    index=index,
                    embedding=text_embeddings,
                    namespace="images"
                )
                image_uuids = [str(uuid4()) for _ in range(len(image_docs))]
                vector_store_images.add_documents(documents=image_docs, ids=image_uuids)
            
            # 3. Store tables
            tables = processed_data.get('tables', [])
            if tables:
                print(f"Storing {len(tables)} tables...")
                
                table_docs = []
                for table_data in tables:
                    doc = Document(
                        page_content=f"Table content:\n{table_data['text_representation']}",
                        metadata={
                            'type': 'table',
                            'page': table_data['page'],
                            'table_index': table_data['table_index'],
                            'markdown': table_data['markdown_representation'],
                            'shape': str(table_data['shape'])
                        }
                    )
                    table_docs.append(doc)
                
                vector_store_tables = PineconeVectorStore(
                    index=index,
                    embedding=text_embeddings,
                    namespace="tables"
                )
                table_uuids = [str(uuid4()) for _ in range(len(table_docs))]
                vector_store_tables.add_documents(documents=table_docs, ids=table_uuids)
            
            print("Multi-modal data stored successfully!")
            
        except Exception as e:
            raise TradingBotException(e, sys)

    def run_pipeline(self, uploaded_files):
        try:
            processed_data = self.load_documents(uploaded_files)
            
            if not any(processed_data.values()):
                print("No valid content found.")
                return
            
            self.store_in_vector_db(processed_data)
            
        except Exception as e:
            raise TradingBotException(e, sys)


if __name__ == '__main__':
    pass
