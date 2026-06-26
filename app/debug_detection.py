import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

from document_scanner import DocumentScanner

def debug_image_processing(image_path: str):
    """Debug document detection step by step."""
    print(f"Debugging image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    print(f"Image shape: {image.shape}")
    print(f"Image size: {image.shape[0] * image.shape[1]} pixels")
    
    # Create debug directory
    debug_dir = Path("debug_output")
    debug_dir.mkdir(exist_ok=True)
    
    # Save original image
    cv2.imwrite(str(debug_dir / "0_original.jpg"), image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(debug_dir / "1_grayscale.jpg"), gray)
    
    # Try different preprocessing steps
    preprocessing_steps = [
        ("Original", gray),
        ("Blur 3x3", cv2.GaussianBlur(gray, (3, 3), 0)),
        ("Blur 5x5", cv2.GaussianBlur(gray, (5, 5), 0)),
        ("Blur 7x7", cv2.GaussianBlur(gray, (7, 7), 0)),
    ]
    
    for name, processed in preprocessing_steps:
        # Edge detection with different thresholds
        for low, high in [(30, 100), (50, 150), (75, 200), (100, 250)]:
            edges = cv2.Canny(processed, low, high)
            
            # Dilate edges
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours on original image
            contour_image = image.copy()
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
            
            # Save results
            filename = f"2_{name}_Canny_{low}_{high}.jpg"
            cv2.imwrite(str(debug_dir / filename), contour_image)
            
            # Check if any contour has 4 corners
            found_document = False
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                
                if len(approx) == 4:
                    area = cv2.contourArea(contour)
                    image_area = image.shape[0] * image.shape[1]
                    
                    if area > image_area * 0.05:
                        print(f"Found document with {len(approx)} corners, area: {area}")
                        
                        # Draw the detected document
                        doc_image = image.copy()
                        cv2.drawContours(doc_image, [approx], -1, (0, 255, 0), 3)
                        for corner in approx.reshape(4, 2):
                            cv2.circle(doc_image, tuple(corner), 10, (0, 0, 255), -1)
                        
                        doc_filename = f"3_DETECTED_{name}_Canny_{low}_{high}.jpg"
                        cv2.imwrite(str(debug_dir / doc_filename), doc_image)
                        found_document = True
                        break
            
            if found_document:
                print(f"Document detected with {name}, Canny {low}-{high}")
    
    # Try adaptive threshold
    for block_size in [11, 15, 21, 31]:
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, block_size, 2)
        cv2.imwrite(str(debug_dir / f"4_adaptive_thresh_{block_size}.jpg"), thresh)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_image = image.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(str(debug_dir / f"5_adaptive_contours_{block_size}.jpg"), contour_image)
    
    # Use the DocumentScanner
    scanner = DocumentScanner()
    result = scanner.detect_document(image)
    
    print(f"\nDetection result: {result.detected}")
    if result.detected:
        print(f"Confidence: {result.confidence}")
        print(f"Rotation angle: {result.rotation_angle}")
        print(f"Processing time: {result.processing_time_ms}ms")
        print(f"Debug info: {result.debug_info}")
        
        # Show final detection
        final_image = scanner.visualize_detection(image, result)
        cv2.imwrite(str(debug_dir / "6_final_detection.jpg"), final_image)
        
        # Try perspective correction
        if result.corners:
            corrected = scanner.correct_perspective(image, result.corners)
            cv2.imwrite(str(debug_dir / "7_corrected.jpg"), corrected)
    else:
        print("No document detected")
        
        # Show contours from the best attempt
        if hasattr(result, 'debug_info'):
            print(f"Debug info: {result.debug_info}")
    
    print(f"\nDebug output saved to: {debug_dir}")
    print("Check the debug_output directory for intermediate images")

def analyze_image_characteristics(image_path: str):
    """Analyze image characteristics to help tune detection parameters."""
    image = cv2.imread(image_path)
    if image is None:
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    print("\n=== Image Analysis ===")
    print(f"Shape: {image.shape}")
    print(f"Mean brightness: {np.mean(gray):.2f}")
    print(f"Std brightness: {np.std(gray):.2f}")
    print(f"Min brightness: {np.min(gray)}")
    print(f"Max brightness: {np.max(gray)}")
    
    # Histogram analysis
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_normalized = hist / hist.sum()
    cumsum = np.cumsum(hist_normalized)
    
    # Find percentiles
    def find_percentile(percent):
        for i, val in enumerate(cumsum):
            if val >= percent / 100.0:
                return i
        return 255
    
    p10 = find_percentile(10)
    p50 = find_percentile(50)
    p90 = find_percentile(90)
    
    print(f"10th percentile: {p10}")
    print(f"50th percentile: {p50}")
    print(f"90th percentile: {p90}")
    print(f"Dynamic range: {p90 - p10}")
    
    # Check if image is too dark or too bright
    if np.mean(gray) < 50:
        print("Warning: Image is very dark")
    elif np.mean(gray) > 200:
        print("Warning: Image is very bright")
    
    # Check if image has good contrast
    if p90 - p10 < 50:
        print("Warning: Image has low contrast")
    
    # Edge analysis
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
    print(f"Edge pixel ratio: {edge_ratio:.4f}")
    
    if edge_ratio < 0.01:
        print("Warning: Very few edges detected")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "samples/ss.jpg"  # Default
    
    # Analyze image
    analyze_image_characteristics(image_path)
    
    # Debug detection
    debug_image_processing(image_path)