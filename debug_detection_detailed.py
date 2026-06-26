import cv2
import numpy as np
import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.document_scanner import DocumentScanner

def debug_detection_detailed(image_path: str):
    """Comprehensive debug for document detection."""
    
    print("=" * 60)
    print(f"Debugging: {image_path}")
    print("=" * 60)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    print(f"Image shape: {image.shape}")
    print(f"Image type: {image.dtype}")
    print(f"Image size: {image.shape[0] * image.shape[1]} pixels")
    
    # Create debug directory
    debug_dir = Path("debug_detailed")
    debug_dir.mkdir(exist_ok=True)
    
    # Save original
    cv2.imwrite(str(debug_dir / "0_original.jpg"), image)
    
    # Test different preprocessing methods
    test_methods = []
    
    # 1. Original image
    test_methods.append(("Original", image))
    
    # 2. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    test_methods.append(("Grayscale", gray))
    
    # 3. Adaptive histogram equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    test_methods.append(("CLAHE", clahe_img))
    
    # 4. Histogram equalization
    eq_img = cv2.equalizeHist(gray)
    test_methods.append(("Histogram Equalization", eq_img))
    
    # 5. Contrast stretching
    p2, p98 = np.percentile(gray, (2, 98))
    stretch_img = cv2.convertScaleAbs(gray, alpha=255/(p98-p2), beta=-p2*255/(p98-p2))
    test_methods.append(("Contrast Stretch", stretch_img))
    
    # 6. Gaussian blur + threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    test_methods.append(("Otsu Threshold", thresh))
    
    # 7. Adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
    test_methods.append(("Adaptive Threshold", adaptive))
    
    # 8. Edge detection
    edges = cv2.Canny(gray, 50, 150)
    test_methods.append(("Canny Edges", edges))
    
    # 9. Edge detection with different thresholds
    edges2 = cv2.Canny(gray, 30, 100)
    test_methods.append(("Canny Edges (30-100)", edges2))
    
    # 10. Edge detection with blur
    blurred_edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    test_methods.append(("Blurred Canny", blurred_edges))
    
    # Test each method with document detection
    scanner = DocumentScanner()
    best_result = None
    best_confidence = 0
    best_method = ""
    
    for method_name, processed_img in test_methods:
        print(f"\nTesting: {method_name}")
        
        # Convert to BGR if grayscale
        if len(processed_img.shape) == 2:
            processed_bgr = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        else:
            processed_bgr = processed_img
        
        # Save processed image
        cv2.imwrite(str(debug_dir / f"1_{method_name.replace(' ', '_')}.jpg"), processed_bgr)
        
        # Detect document
        result = scanner.detect_document(processed_bgr)
        
        print(f"  Detected: {result.detected}")
        if result.detected:
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Corners: {result.corners}")
            print(f"  Rotation: {result.rotation_angle:.1f}°")
            print(f"  Method: {result.debug_info.get('contour_method', 'unknown')}")
            
            if result.confidence > best_confidence:
                best_confidence = result.confidence
                best_result = result
                best_method = method_name
                
            # Visualize detection
            vis_img = scanner.visualize_detection(processed_bgr, result)
            cv2.imwrite(str(debug_dir / f"2_{method_name.replace(' ', '_')}_detected.jpg"), vis_img)
        else:
            print(f"  Debug info: {result.debug_info}")
    
    # Show best result
    print("\n" + "=" * 60)
    print("BEST RESULT:")
    print("=" * 60)
    if best_result:
        print(f"Method: {best_method}")
        print(f"Confidence: {best_confidence:.3f}")
        print(f"Detected: {best_result.detected}")
        if best_result.detected:
            print(f"Corners: {best_result.corners}")
            
            # Try perspective correction with best result
            if best_result.corners:
                corrected = scanner.correct_perspective(image, best_result.corners)
                cv2.imwrite(str(debug_dir / "3_best_corrected.jpg"), corrected)
                print("Corrected image saved to debug_detailed/3_best_corrected.jpg")
    else:
        print("No document detected in any method")
    
    print(f"\nDebug output saved to: {debug_dir}")
    
    return best_result

def analyze_image_for_detection(image_path: str):
    """Analyze image characteristics that might affect detection."""
    
    image = cv2.imread(image_path)
    if image is None:
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    print("\n" + "=" * 60)
    print("IMAGE ANALYSIS:")
    print("=" * 60)
    
    # Basic statistics
    print(f"Shape: {image.shape}")
    print(f"Mean: {np.mean(gray):.2f}")
    print(f"Std: {np.std(gray):.2f}")
    print(f"Min: {np.min(gray)}")
    print(f"Max: {np.max(gray)}")
    print(f"Dynamic Range: {np.max(gray) - np.min(gray)}")
    
    # Histogram analysis
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist / hist.sum()
    cumsum = np.cumsum(hist_norm)
    
    # Percentiles
    def find_percentile(p):
        for i, val in enumerate(cumsum):
            if val >= p / 100.0:
                return i
        return 255
    
    p1 = find_percentile(1)
    p5 = find_percentile(5)
    p10 = find_percentile(10)
    p25 = find_percentile(25)
    p50 = find_percentile(50)
    p75 = find_percentile(75)
    p90 = find_percentile(90)
    p95 = find_percentile(95)
    p99 = find_percentile(99)
    
    print(f"\nPercentiles:")
    print(f"  1%: {p1}")
    print(f"  5%: {p5}")
    print(f"  10%: {p10}")
    print(f"  25%: {p25}")
    print(f"  50%: {p50}")
    print(f"  75%: {p75}")
    print(f"  90%: {p90}")
    print(f"  95%: {p95}")
    print(f"  99%: {p99}")
    
    # Image quality indicators
    dynamic_range = p99 - p1
    print(f"\nImage Quality:")
    print(f"  Dynamic Range (1%-99%): {dynamic_range}")
    print(f"  Mean Brightness: {p50}")
    
    if dynamic_range < 100:
        print("  ⚠️ Low contrast image")
    elif dynamic_range < 150:
        print("  ⚠️ Medium contrast image")
    else:
        print("  ✅ Good contrast image")
    
    if p50 < 50:
        print("  ⚠️ Very dark image")
    elif p50 < 100:
        print("  ⚠️ Dark image")
    elif p50 < 150:
        print("  ✅ Normal brightness")
    elif p50 < 200:
        print("  ⚠️ Bright image")
    else:
        print("  ⚠️ Very bright image")
    
    # Edge analysis
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
    print(f"  Edge Ratio: {edge_ratio:.4f}")
    
    if edge_ratio < 0.005:
        print("  ⚠️ Very few edges detected")
    elif edge_ratio < 0.02:
        print("  ⚠️ Low edge density")
    else:
        print("  ✅ Good edge density")
    
    # Color analysis (if image is color)
    if len(image.shape) == 3:
        # Check if image is grayscale disguised as color
        b, g, r = cv2.split(image)
        b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
        
        print(f"\nColor Analysis:")
        print(f"  B mean: {b_mean:.2f}")
        print(f"  G mean: {g_mean:.2f}")
        print(f"  R mean: {r_mean:.2f}")
        
        # Check if it's mostly grayscale
        color_variance = np.std([b_mean, g_mean, r_mean])
        if color_variance < 5:
            print("  ✅ Image appears to be grayscale (BGR values similar)")
        else:
            print("  📸 Image has color information")
    
    # Save histogram for visualization
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.hist(gray.ravel(), bins=256, range=[0, 256])
    plt.title("Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.axvline(p10, color='r', linestyle='--', label='10%')
    plt.axvline(p50, color='g', linestyle='--', label='50%')
    plt.axvline(p90, color='b', linestyle='--', label='90%')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("debug_detailed/histogram_analysis.png")
    plt.close()
    
    print("\nHistogram saved to debug_detailed/histogram_analysis.png")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "samples/business_card_normal.jpg"
    
    # Analyze image
    analyze_image_for_detection(image_path)
    
    # Debug detection
    result = debug_detection_detailed(image_path)
    
    print("\n" + "=" * 60)
    print("SUGGESTIONS:")
    print("=" * 60)
    if result and result.detected:
        print("✅ Document detected successfully!")
        print(f"Confidence: {result.confidence:.3f}")
    else:
        print("❌ Document detection failed. Try:")
        print("1. Check if the document has clear edges")
        print("2. Ensure good contrast between document and background")
        print("3. Try using a different image")
        print("4. Check the debug output in debug_detailed/ directory")