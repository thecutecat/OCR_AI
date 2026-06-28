import cv2
import numpy as np
import os
from pathlib import Path
from app.document_scanner import DocumentScanner
from app.image_processor import ImageEnhancer
from app.ocr_engine import OCREngine

def create_test_borderless_image():
    """Create a test image with text but no borders."""
    # Create white background
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Add text
    cv2.putText(img, "This is a test document", (100, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.putText(img, "with no clear borders", (100, 220), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(img, "The system should detect", (100, 290), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(img, "the text region automatically", (100, 360), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    return img

def test_borderless_detection():
    """Test detection on borderless image."""
    print("=" * 60)
    print("Testing Borderless Document Detection")
    print("=" * 60)
    
    # Create test image
    image = create_test_borderless_image()
    output_path = "samples/borderless_test.jpg"
    cv2.imwrite(output_path, image)
    print(f"✓ Created test image: {output_path}")
    
    # Initialize scanner
    scanner = DocumentScanner()
    
    # Detect document
    print("\n🔍 Running detection...")
    result = scanner.detect_document(image)
    
    print(f"\n📊 Detection Result:")
    print(f"  Detected: {result.detected}")
    if result.detected:
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Corners: {result.corners}")
        print(f"  Debug Info:")
        for key, value in result.debug_info.items():
            print(f"    {key}: {value}")
        
        # Visualize
        vis_image = scanner.visualize_detection(image, result)
        cv2.imwrite("samples/borderless_detected.jpg", vis_image)
        print(f"\n✓ Visualization saved: samples/borderless_detected.jpg")
        
        # Test perspective correction
        if result.corners:
            corrected = scanner.correct_perspective(image, result.corners)
            cv2.imwrite("samples/borderless_corrected.jpg", corrected)
            print(f"✓ Corrected image saved: samples/borderless_corrected.jpg")
    else:
        print("  ✗ Detection failed")
        print(f"  Debug Info: {result.debug_info}")
    
    return result

if __name__ == "__main__":
    test_borderless_detection()