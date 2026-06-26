import cv2
import numpy as np
from app.document_scanner_enhanced import DocumentScannerEnhanced

scanner = DocumentScannerEnhanced()
image = np.zeros((600, 800, 3), dtype=np.uint8)

# Use fixed seed to match the test environment
np.random.seed(42) # Wait, is seed set in the test? No, the test doesn't seed for lines, but let's check
for _ in range(50):
    x1, y1 = np.random.randint(0, 800), np.random.randint(0, 600)
    x2, y2 = np.random.randint(0, 800), np.random.randint(0, 600)
    cv2.line(image, (x1, y1), (x2, y2),
            (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)),
            1)

result = scanner.detect_document(image)
print("Detected:", result.detected)
print("Corners:", result.corners)
print("Confidence:", result.confidence)
print("Debug info:", result.debug_info)
