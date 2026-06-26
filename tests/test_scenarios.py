import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now imports will work
import pytest
import cv2
import numpy as np
from pathlib import Path
import json

#from app.document_scanner import DocumentScanner
from app.document_scanner_enhanced import DocumentScannerEnhanced as DocumentScanner
from app.image_processor import ImageEnhancer
from app.ocr_engine import OCREngine

class TestDocumentScanner:
    @pytest.fixture
    def scanner(self):
        return DocumentScanner()
    
    @pytest.fixture
    def enhancer(self):
        return ImageEnhancer()
    
    def test_rotated_document(self, scanner):
        """Test rotation handling"""
        # Create a rotated document image
        image = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.rectangle(image, (200, 200), (600, 400), (255, 255, 255), -1)
        # Rotate the image
        center = (400, 300)
        matrix = cv2.getRotationMatrix2D(center, 15, 1.0)
        rotated = cv2.warpAffine(image, matrix, (800, 600))
        
        result = scanner.detect_document(rotated)
        assert result.detected
        assert abs(result.rotation_angle) > 0
    
    def test_low_light_image(self, scanner, enhancer):
        """Test low light handling"""
        # Create a low light image
        image = np.ones((600, 800, 3), dtype=np.uint8) * 30
        cv2.rectangle(image, (200, 200), (600, 400), (50, 50, 50), -1)
        
        # Enhance image
        enhanced = enhancer.enhance(image)
        result = scanner.detect_document(enhanced)
        
        # Should still detect document
        assert result.detected
    
    def test_multiple_objects(self, scanner):
        """Test correct document selection with multiple objects"""
        image = np.zeros((600, 800, 3), dtype=np.uint8)
        # Add multiple rectangles
        cv2.rectangle(image, (50, 50), (150, 150), (255, 255, 255), -1)  # Small
        cv2.rectangle(image, (200, 200), (600, 400), (255, 255, 255), -1)  # Large (document)
        cv2.rectangle(image, (650, 500), (750, 550), (255, 255, 255), -1)  # Small
        
        result = scanner.detect_document(image)
        assert result.detected
        # The largest rectangle should be selected
        assert len(result.corners) == 4
        
    def test_no_document(self, scanner):
        """Test graceful failure when no document present"""
        # Image with no clear document
        image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        
        result = scanner.detect_document(image)
        assert not result.detected
    
    def test_noisy_image(self, scanner, enhancer):
        """Test stable detection in noisy images"""
        # Create a document image
        image = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.rectangle(image, (200, 200), (600, 400), (255, 255, 255), -1)
        
        # Add salt and pepper noise
        noise = np.random.random(image.shape) > 0.95
        image[noise] = np.random.randint(0, 255, image[noise].shape)
        
        enhanced = enhancer.denoise(image)
        result = scanner.detect_document(enhanced)
        assert result.detected
    
    def test_partial_shadow(self, scanner):
        """Test OCR still functional with partial shadow"""
        # Create an image with gradient shadow
        image = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.rectangle(image, (200, 200), (600, 400), (255, 255, 255), -1)
        
        # Apply gradient shadow
        for i in range(400, 600):
            image[i, 200:600] = image[i, 200:600] * (1 - (i - 400) / 200)
        
        result = scanner.detect_document(image)
        assert result.detected

class TestOCR:
    @pytest.fixture
    def ocr_engine(self):
        # Skip if Tesseract is not available
        try:
            return OCREngine(language='eng', document_type='business_card')
        except Exception as e:
            pytest.skip(f"Tesseract not available: {e}")
    
    @pytest.fixture
    def enhancer(self):
        return ImageEnhancer()
    
    def test_ocr_accuracy(self, ocr_engine, enhancer):
        """Test OCR accuracy on enhanced images"""
        # Create a simple test image with text
        image = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(image, "John Doe", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(image, "john@example.com", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        enhanced = enhancer.enhance(image)
        result = ocr_engine.extract_text(enhanced)
        
        assert 'John Doe' in result['raw_text'] or 'John' in result['raw_text']
        assert result['confidence'] > 0.3

# Run tests with coverage
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])