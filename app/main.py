import cv2
import numpy as np
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

#from document_scanner import DocumentScanner
from document_scanner_enhanced import DocumentScannerEnhanced as DocumentScanner
from image_processor import ImageEnhancer
from ocr_engine import OCREngine
from metadata_generator import MetadataGenerator
from database import DatabaseManager

class DocumentScannerApp:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scanner = DocumentScanner()
        self.enhancer = ImageEnhancer()
        self.ocr = OCREngine(
            language=config.get('ocr_language', 'eng'),
            document_type=config.get('document_type', 'business_card')
        )
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
                    'error': 'No document detected',
                    'processing_time_ms': (time.time() - start_time) * 1000
                }
            
            # Visualize detection
            debug_image = self.scanner.visualize_detection(image, detection_result)
            debug_path = self._generate_output_path('debug', image_path)
            cv2.imwrite(debug_path, debug_image)
            self.logger.info(f"Debug image saved to: {debug_path}")
            
            # Step 2: Perspective Correction
            corrected_image = self.scanner.correct_perspective(image, detection_result.corners)
            corrected_path = self._generate_output_path('processed', image_path, 'corrected')
            cv2.imwrite(corrected_path, corrected_image)
            self.logger.info(f"Corrected image saved to: {corrected_path}")
            
            # Step 3: Image Enhancement
            enhanced_image = self.enhancer.enhance(corrected_image, return_color=True)
            enhanced_path = self._generate_output_path('processed', image_path, 'enhanced')
            cv2.imwrite(enhanced_path, enhanced_image)
            self.logger.info(f"Enhanced image saved to: {enhanced_path}")
            
            # Step 4: OCR Extraction
            ocr_result = self.ocr.extract_text(enhanced_image)
            
            # Step 5: Generate Metadata
            processing_time = (time.time() - start_time) * 1000
            metadata = self.metadata_generator.generate_metadata(
                image, detection_result, ocr_result, processing_time
            )
            
            # Step 6: Save JSON output
            json_path = self._generate_output_path('json', image_path)
            self.metadata_generator.save_metadata(metadata, json_path)
            
            # Step 7: Save to database if configured
            if self.db:
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
            self.logger.error(f"Error processing image: {e}")
            return {
                'success': False,
                'error': str(e),
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