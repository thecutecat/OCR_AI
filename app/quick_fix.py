import cv2
import numpy as np
import os
from app.document_scanner import DocumentScanner

def quick_test():
    """Quick test for document detection."""
    # Create scanner
    scanner = DocumentScanner()
    
    # Test with sample image
    image_path = "samples/ss.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Creating a test image...")
        from create_test_image import create_business_card
        test_image = create_business_card()
        image_path = "samples/test_created.jpg"
        cv2.imwrite(image_path, cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
        print(f"Created test image: {image_path}")
    
    # Load and process image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"Image loaded: {image.shape}")
    
    # Detect document
    result = scanner.detect_document(image)
    
    print(f"\nDetection Result:")
    print(f"  Detected: {result.detected}")
    
    if result.detected:
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Rotation angle: {result.rotation_angle:.1f}°")
        print(f"  Processing time: {result.processing_time_ms:.1f}ms")
        print(f"  Corners: {result.corners}")
        print(f"  Debug info: {result.debug_info}")
        
        # Visualize and save
        vis_image = scanner.visualize_detection(image, result)
        output_path = "detection_result.jpg"
        cv2.imwrite(output_path, vis_image)
        print(f"\nVisualization saved to: {output_path}")
        
        # Try perspective correction
        if result.corners:
            corrected = scanner.correct_perspective(image, result.corners)
            output_path = "corrected_result.jpg"
            cv2.imwrite(output_path, corrected)
            print(f"Corrected image saved to: {output_path}")
    else:
        print(f"  Error: {result.debug_info}")
    
    return result

if __name__ == "__main__":
    quick_test()