import cv2
import numpy as np
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
import sys

from document_scanner import DocumentScanner
from image_processor import ImageEnhancer
from ocr_engine import OCREngine
from metadata_generator import MetadataGenerator
from database import DatabaseManager

class DocumentScannerApp:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scanner = DocumentScanner()
        self.enhancer = ImageEnhancer()
        
        # Initialize OCR with Tesseract path from config
        tesseract_path = config.get('tesseract_path')
        try:
            self.ocr = OCREngine(
                language=config.get('ocr_language', 'eng'),
                document_type=config.get('document_type', 'business_card'),
                tesseract_path=tesseract_path
            )
        except RuntimeError as e:
            self.logger.error(f"OCR initialization failed: {e}")
            self.logger.error("Please install Tesseract OCR and make sure it's in your PATH")
            self.logger.error("Download from: https://github.com/UB-Mannheim/tesseract/wiki")
            raise
        
        self.metadata_generator = MetadataGenerator()
        
        # Initialize database if configured
        self.db = None
        if config.get('use_database', False):
            self.db = DatabaseManager(config.get('db_connection_string', ''))
        
        # Setup logging
        self._setup_logging()
        
        # Create output directories
        self._create_directories()
    
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('document_scanner.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _create_directories(self):
        """Create necessary output directories."""
        dirs = [
            'outputs/debug',
            'outputs/processed',
            'outputs/json'
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single image through the complete pipeline.
        """
        start_time = time.time()
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            self.logger.info(f"Processing image: {image_path}")
            
            # Step 1: Document Detection
            detection_result = self.scanner.detect_document(image)
            
            if not detection_result.detected:
                self.logger.warning("No document detected in the image")
                return {
                    'success': False,
                    'error': 'No document detected. The image may not contain a clear document or the document is too blurry. Please ensure the document is clearly visible.',
                    'processing_time_ms': (time.time() - start_time) * 1000,
                    'debug_info': detection_result.debug_info
                }
            
            # Visualize detection
            debug_image = self.scanner.visualize_detection(image, detection_result)
            debug_path = self._generate_output_path('debug', image_path)
            cv2.imwrite(debug_path, debug_image)
            self.logger.info(f"Debug image saved to: {debug_path}")
            
            # Step 2: Perspective Correction
            try:
                corrected_image = self.scanner.correct_perspective(image, detection_result.corners)
                corrected_path = self._generate_output_path('processed', image_path, 'corrected')
                cv2.imwrite(corrected_path, corrected_image)
                self.logger.info(f"Corrected image saved to: {corrected_path}")
            except Exception as e:
                self.logger.error(f"Perspective correction failed: {e}")
                return {
                    'success': False,
                    'error': f'Perspective correction failed: {str(e)}. The detected corners may be invalid.',
                    'processing_time_ms': (time.time() - start_time) * 1000,
                    'debug_info': detection_result.debug_info
                }
            
            # Step 3: Image Enhancement
            try:
                enhanced_image = self.enhancer.enhance(corrected_image, return_color=True)
                enhanced_path = self._generate_output_path('processed', image_path, 'enhanced')
                cv2.imwrite(enhanced_path, enhanced_image)
                self.logger.info(f"Enhanced image saved to: {enhanced_path}")
            except Exception as e:
                self.logger.error(f"Image enhancement failed: {e}")
                return {
                    'success': False,
                    'error': f'Image enhancement failed: {str(e)}',
                    'processing_time_ms': (time.time() - start_time) * 1000,
                    'debug_info': detection_result.debug_info
                }
            
            # Step 4: OCR Extraction
            try:
                # Convert enhanced color image to grayscale for OCR
                if len(enhanced_image.shape) == 3:
                    enhanced_gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
                else:
                    enhanced_gray = enhanced_image
                
                ocr_result = self.ocr.extract_text(enhanced_gray)
            except Exception as e:
                self.logger.error(f"OCR extraction failed: {e}")
                return {
                    'success': False,
                    'error': f'OCR extraction failed: {str(e)}. Please ensure Tesseract is installed.',
                    'processing_time_ms': (time.time() - start_time) * 1000,
                    'debug_info': detection_result.debug_info
                }
            
            # Step 5: Generate Metadata
            try:
                processing_time = (time.time() - start_time) * 1000
                metadata = self.metadata_generator.generate_metadata(
                    image, detection_result, ocr_result, processing_time
                )
            except Exception as e:
                self.logger.error(f"Metadata generation failed: {e}")
                return {
                    'success': False,
                    'error': f'Metadata generation failed: {str(e)}',
                    'processing_time_ms': (time.time() - start_time) * 1000,
                    'debug_info': detection_result.debug_info
                }
            
            # Step 6: Save JSON output
            try:
                json_path = self._generate_output_path('json', image_path)
                self.metadata_generator.save_metadata(metadata, json_path)
            except Exception as e:
                self.logger.error(f"JSON save failed: {e}")
                # Continue even if JSON save fails
            
            # Step 7: Save to database if configured
            if self.db:
                try:
                    document_id = metadata.get('document_id', '')
                    self.db.save_document(
                        document_id=document_id,
                        document_type=metadata.get('document_type', ''),
                        image_path=image_path,
                        processed_image_path=enhanced_path,
                        metadata=metadata,
                        ocr_result=ocr_result
                    )
                    
                    self.db.save_processing_log(
                        document_id=document_id,
                        processing_time_ms=processing_time,
                        ocr_confidence=metadata['ocr_confidence'],
                        success=True
                    )
                except Exception as e:
                    self.logger.error(f"Database save failed: {e}")
                    # Continue even if database save fails
            
            result = {
                'success': True,
                'document_id': metadata.get('document_id'),
                'metadata': metadata,
                'ocr_result': ocr_result,
                'processing_time_ms': processing_time,
                'output_files': {
                    'debug': debug_path,
                    'corrected': corrected_path,
                    'enhanced': enhanced_path,
                    'json': json_path
                }
            }
            
            self.logger.info(f"Successfully processed image in {processing_time:.2f}ms")
            return result
            
        except Exception as e:
            error_message = str(e)
            self.logger.error(f"Error processing image: {error_message}")
            
            return {
                'success': False,
                'error': f'Processing failed: {error_message}',
                'processing_time_ms': (time.time() - start_time) * 1000
            }
    
    def _generate_output_path(self, output_type: str, image_path: str, suffix: Optional[str] = None) -> str:
        """Generate output path for processed images."""
        filename = Path(image_path).stem
        extension = Path(image_path).suffix
        
        if suffix:
            output_filename = f"{filename}_{suffix}{extension}"
        else:
            output_filename = f"{filename}{extension}"
        
        return os.path.join('outputs', output_type, output_filename)