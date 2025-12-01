"""
Enhanced Multi-Modal Document Processor with OCR Support
Adds Tesseract OCR for scanned documents and images
"""

import os
import tempfile
from typing import List, Dict, Any, Optional
from PIL import Image
import pytesseract
import pymupdf
import io
import base64
import json
from pathlib import Path
import numpy as np
from pdf2image import convert_from_path
import cv2

from utils.model_loaders import ModelLoader
from exception.exceptions import TradingBotException
from custom_logging.my_logger import logger
import sys


class OCRProcessor:
    """
    Handles OCR processing for scanned documents and images
    """
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize OCR Processor
        
        Args:
            tesseract_cmd: Path to tesseract executable (optional)
        """
        logger.info("Initializing OCR Processor...")
        
        # Set Tesseract path if provided
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        # Test Tesseract installation
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
        except Exception as e:
            logger.error(f"Tesseract not found: {e}")
            raise TradingBotException(
                "Tesseract OCR not installed. Install with: sudo apt-get install tesseract-ocr",
                sys
            )
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR results
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed PIL Image
        """
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply thresholding for better text recognition
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
            
            # Convert back to PIL Image
            return Image.fromarray(denoised)
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed, using original: {e}")
            return image
    
    def extract_text_from_image(
        self, 
        image: Image.Image, 
        preprocess: bool = True,
        lang: str = 'eng'
    ) -> Dict[str, Any]:
        """
        Extract text from image using Tesseract OCR
        
        Args:
            image: PIL Image
            preprocess: Whether to preprocess image
            lang: Language for OCR (default: 'eng')
            
        Returns:
            Dictionary with extracted text and confidence scores
        """
        try:
            # Preprocess if requested
            if preprocess:
                image = self.preprocess_image(image)
            
            # Extract text with detailed data
            ocr_data = pytesseract.image_to_data(
                image, 
                lang=lang, 
                output_type=pytesseract.Output.DICT
            )
            
            # Extract plain text
            text = pytesseract.image_to_string(image, lang=lang)
            
            # Calculate average confidence
            confidences = [int(conf) for conf in ocr_data['conf'] if conf != '-1']
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            logger.info(f"OCR extracted {len(text)} characters with {avg_confidence:.2f}% confidence")
            
            return {
                'text': text.strip(),
                'confidence': avg_confidence,
                'word_count': len(text.split()),
                'detailed_data': ocr_data
            }
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            raise TradingBotException(e, sys)
    
    def process_scanned_pdf(
        self, 
        pdf_path: str, 
        dpi: int = 300
    ) -> List[Dict[str, Any]]:
        """
        Process scanned PDF using OCR
        
        Args:
            pdf_path: Path to PDF file
            dpi: DPI for PDF to image conversion
            
        Returns:
            List of page data with OCR text
        """
        try:
            logger.info(f"Processing scanned PDF: {pdf_path}")
            
            # Convert PDF pages to images
            images = convert_from_path(pdf_path, dpi=dpi)
            
            pages_data = []
            
            for page_num, image in enumerate(images, start=1):
                logger.info(f"Processing page {page_num}/{len(images)}")
                
                # Extract text via OCR
                ocr_result = self.extract_text_from_image(image)
                
                pages_data.append({
                    'page_num': page_num,
                    'text': ocr_result['text'],
                    'confidence': ocr_result['confidence'],
                    'word_count': ocr_result['word_count'],
                    'is_ocr': True
                })
            
            logger.info(f"Extracted text from {len(pages_data)} pages via OCR")
            
            return pages_data
            
        except Exception as e:
            logger.error(f"Scanned PDF processing failed: {e}")
            raise TradingBotException(e, sys)
    
    def is_pdf_scanned(self, pdf_path: str) -> bool:
        """
        Detect if PDF is scanned (image-based) or text-based
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            True if scanned, False if text-based
        """
        try:
            pdf_document = pymupdf.open(pdf_path)
            
            # Check first few pages
            pages_to_check = min(3, len(pdf_document))
            text_chars = 0
            
            for page_num in range(pages_to_check):
                page = pdf_document[page_num]
                text = page.get_text()
                text_chars += len(text.strip())
            
            pdf_document.close()
            
            # If very little text found, likely scanned
            is_scanned = text_chars < 100
            
            logger.info(f"PDF detection: {'SCANNED' if is_scanned else 'TEXT-BASED'} ({text_chars} chars)")
            
            return is_scanned
            
        except Exception as e:
            logger.warning(f"PDF type detection failed: {e}")
            return False


class EnhancedMultiModalProcessor:
    """
    Enhanced processor with OCR support
    """
    
    def __init__(self, enable_ocr: bool = True):
        logger.info("Initializing Enhanced Multi-Modal Processor...")
        
        # Import original processor components
        from data_ingestion.multimodal_ingestion import MultiModalProcessor
        self.base_processor = MultiModalProcessor()
        
        # Initialize OCR
        self.enable_ocr = enable_ocr
        if enable_ocr:
            try:
                self.ocr_processor = OCRProcessor()
                logger.info("OCR support enabled")
            except Exception as e:
                logger.warning(f"OCR initialization failed: {e}")
                self.enable_ocr = False
        
        logger.info("Enhanced Multi-Modal Processor initialized")
    
    def extract_text_from_pdf(
        self, 
        pdf_path: str, 
        auto_detect_scanned: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Extract text from PDF with automatic OCR fallback
        
        Args:
            pdf_path: Path to PDF
            auto_detect_scanned: Auto-detect if PDF is scanned
            
        Returns:
            List of page data with text
        """
        try:
            # Check if PDF is scanned
            is_scanned = False
            if auto_detect_scanned and self.enable_ocr:
                is_scanned = self.ocr_processor.is_pdf_scanned(pdf_path)
            
            if is_scanned:
                logger.info("Using OCR for scanned PDF")
                return self.ocr_processor.process_scanned_pdf(pdf_path)
            else:
                logger.info("Extracting text from text-based PDF")
                return self._extract_text_native(pdf_path)
                
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise TradingBotException(e, sys)
    
    def _extract_text_native(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text using native PDF text extraction"""
        try:
            pdf_document = pymupdf.open(pdf_path)
            pages_data = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                text = page.get_text()
                
                pages_data.append({
                    'page_num': page_num + 1,
                    'text': text,
                    'confidence': 100.0,  # Native extraction
                    'word_count': len(text.split()),
                    'is_ocr': False
                })
            
            pdf_document.close()
            
            return pages_data
            
        except Exception as e:
            logger.error(f"Native text extraction failed: {e}")
            raise TradingBotException(e, sys)
    
    def process_image_file(
        self, 
        image_path: str
    ) -> Dict[str, Any]:
        """
        Process standalone image file with OCR and captioning
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with OCR text, caption, and embedding
        """
        try:
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            result = {
                'image_path': image_path,
                'format': image.format,
                'size': image.size
            }
            
            # OCR text extraction
            if self.enable_ocr:
                ocr_result = self.ocr_processor.extract_text_from_image(image)
                result['ocr_text'] = ocr_result['text']
                result['ocr_confidence'] = ocr_result['confidence']
            
            # Image captioning
            caption = self.base_processor.generate_image_caption(image)
            result['caption'] = caption
            
            # Image embedding
            embedding = self.base_processor.generate_image_embedding(image)
            result['embedding'] = embedding
            
            logger.info(f"Processed image: {image_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            raise TradingBotException(e, sys)


# Export enhanced processor
__all__ = ['EnhancedMultiModalProcessor', 'OCRProcessor']
