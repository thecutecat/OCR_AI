import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging
from dataclasses import dataclass
import time

@dataclass
class DocumentDetectionResult:
    detected: bool
    contours: Optional[np.ndarray]
    corners: Optional[List[Tuple[int, int]]]
    rotation_angle: float
    processing_time_ms: float
    confidence: float
    debug_info: Dict[str, any]

class DocumentScannerEnhanced:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def _ensure_color_image(self, image: np.ndarray) -> np.ndarray:
        """Ensure image is 3-channel BGR."""
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            return image
        else:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    def _preprocess_image(self, image: np.ndarray) -> List[np.ndarray]:
        """Generate multiple preprocessing variants for better detection."""
        variants = []
        
        # Original
        variants.append(("original", image))
        
        # Convert to grayscale and back to BGR
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        variants.append(("clahe", cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)))
        
        # 2. Histogram Equalization
        eq_img = cv2.equalizeHist(gray)
        variants.append(("equalized", cv2.cvtColor(eq_img, cv2.COLOR_GRAY2BGR)))
        
        # 3. Contrast Stretching
        p2, p98 = np.percentile(gray, (2, 98))
        stretch_img = cv2.convertScaleAbs(gray, alpha=255/(p98-p2), beta=-p2*255/(p98-p2))
        variants.append(("stretched", cv2.cvtColor(stretch_img, cv2.COLOR_GRAY2BGR)))
        
        # 4. Gamma correction (brighten dark images)
        gamma = 1.5
        gamma_corrected = np.power(gray / 255.0, gamma) * 255
        gamma_corrected = gamma_corrected.astype(np.uint8)
        variants.append(("gamma", cv2.cvtColor(gamma_corrected, cv2.COLOR_GRAY2BGR)))
        
        # 5. Gaussian blur (reduce noise)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        variants.append(("blurred", cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)))
        
        # 6. Sharpened
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        variants.append(("sharpened", cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)))
        
        # 7. Bilateral filter (preserve edges)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        variants.append(("bilateral", cv2.cvtColor(bilateral, cv2.COLOR_GRAY2BGR)))
        
        return variants
    
    def detect_document(self, image: np.ndarray) -> DocumentDetectionResult:
        """
        Detect document using multiple preprocessing and detection strategies.
        """
        start_time = time.time()
        debug_info = {}
        
        # Ensure image is color
        try:
            image = self._ensure_color_image(image)
            debug_info['input_format'] = 'bgr'
        except Exception as e:
            self.logger.warning(f"Failed to convert image: {e}")
            return DocumentDetectionResult(
                detected=False,
                contours=None,
                corners=None,
                rotation_angle=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                confidence=0.0,
                debug_info={'error': str(e)}
            )
        
        # Try different preprocessing variants
        variants = self._preprocess_image(image)
        
        best_result = None
        best_confidence = 0.0
        
        for variant_name, variant_img in variants:
            # Try detection on this variant
            result = self._detect_single_variant(variant_img, variant_name, debug_info)
            
            if result['detected'] and result['confidence'] > best_confidence:
                best_confidence = result['confidence']
                best_result = result
                debug_info['best_variant'] = variant_name
        
        # If no detection, try adaptive threshold-based detection
        if not best_result:
            debug_info['fallback_used'] = True
            best_result = self._detect_with_adaptive_threshold(image, debug_info)
        
        # If still no detection, try edge-based with different parameters
        if not best_result or not best_result['detected']:
            debug_info['edge_fallback'] = True
            best_result = self._detect_with_edges_only(image, debug_info)
        
        processing_time = (time.time() - start_time) * 1000
        
        if best_result and best_result['detected']:
            confidence = min(1.0, max(0.0, best_result.get('confidence', 0.0)))
            
            return DocumentDetectionResult(
                detected=True,
                contours=best_result.get('contours'),
                corners=best_result.get('corners'),
                rotation_angle=best_result.get('rotation_angle', 0.0),
                processing_time_ms=processing_time,
                confidence=confidence,
                debug_info=debug_info
            )
        else:
            return DocumentDetectionResult(
                detected=False,
                contours=None,
                corners=None,
                rotation_angle=0.0,
                processing_time_ms=processing_time,
                confidence=0.0,
                debug_info=debug_info
            )
    
    def _detect_single_variant(self, image: np.ndarray, variant_name: str, debug_info: Dict) -> Dict:
        """Try detection on a single image variant."""
        
        # Resize for faster processing
        height, width = image.shape[:2]
        scale = min(800 / width, 800 / height) if max(width, height) > 800 else 1.0
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Try multiple edge detection parameters
        edge_params = [
            (30, 100),
            (50, 150),
            (75, 200),
            (100, 250),
            (20, 80),
            (40, 120)
        ]
        
        for canny_low, canny_high in edge_params:
            edges = cv2.Canny(gray, canny_low, canny_high)
            
            # Dilate to close gaps
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                continue
            
            # Sort by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            for contour in contours[:15]:
                perimeter = cv2.arcLength(contour, True)
                
                # Try different epsilon values
                for epsilon in [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]:
                    approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
                    
                    if len(approx) == 4:
                        area = cv2.contourArea(contour)
                        image_area = image.shape[0] * image.shape[1]
                        
                        # Check if area is reasonable (between 3% and 95% of image)
                        if area > image_area * 0.03 and area < image_area * 0.95:
                            corners = approx.reshape(4, 2).tolist()
                            
                            # Validate rectangle
                            if self._is_valid_rectangle(corners):
                                # Calculate confidence
                                confidence = self._calculate_confidence(corners, area, image_area)
                                rotation_angle = self._calculate_rotation(corners)
                                
                                debug_info[f'{variant_name}_detected'] = True
                                debug_info[f'{variant_name}_confidence'] = confidence
                                
                                # Scale corners back
                                if scale < 1.0:
                                    scaled_corners = [[int(c[0] / scale), int(c[1] / scale)] for c in corners]
                                else:
                                    scaled_corners = corners
                                
                                return {
                                    'detected': True,
                                    'contours': contour,
                                    'corners': scaled_corners,
                                    'rotation_angle': rotation_angle,
                                    'confidence': confidence,
                                    'variant': variant_name,
                                    'epsilon': epsilon
                                }
        
        debug_info[f'{variant_name}_detected'] = False
        return {'detected': False}
    
    def _detect_with_adaptive_threshold(self, image: np.ndarray, debug_info: Dict) -> Dict:
        """Fallback detection using adaptive thresholding."""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Try different block sizes
            for block_size in [11, 15, 21, 31, 41]:
                thresh = cv2.adaptiveThreshold(
                    gray, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, block_size, 2
                )
                
                # Morphological operations
                kernel = np.ones((3, 3), np.uint8)
                closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
                opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
                
                contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                    continue
                
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                for contour in contours[:10]:
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                    
                    if len(approx) == 4:
                        area = cv2.contourArea(contour)
                        image_area = image.shape[0] * image.shape[1]
                        
                        if area > image_area * 0.03 and area < image_area * 0.95:
                            corners = approx.reshape(4, 2).tolist()
                            
                            if self._is_valid_rectangle(corners):
                                confidence = self._calculate_confidence(corners, area, image_area)
                                rotation_angle = self._calculate_rotation(corners)
                                
                                debug_info['adaptive_threshold_success'] = True
                                debug_info['adaptive_block_size'] = block_size
                                
                                return {
                                    'detected': True,
                                    'contours': contour,
                                    'corners': corners,
                                    'rotation_angle': rotation_angle,
                                    'confidence': confidence
                                }
            
            debug_info['adaptive_threshold_success'] = False
            return {'detected': False}
            
        except Exception as e:
            debug_info['adaptive_threshold_error'] = str(e)
            return {'detected': False}
    
    def _detect_with_edges_only(self, image: np.ndarray, debug_info: Dict) -> Dict:
        """Last resort: detect using only edge information."""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Try many edge detection parameters
            for low in [20, 30, 40, 50, 60, 70]:
                for high in [80, 100, 120, 150, 180, 200]:
                    if low >= high:
                        continue
                        
                    edges = cv2.Canny(gray, low, high)
                    
                    # Dilate more to connect edges
                    kernel = np.ones((7, 7), np.uint8)
                    dilated = cv2.dilate(edges, kernel, iterations=3)
                    
                    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if not contours:
                        continue
                    
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    
                    for contour in contours[:15]:
                        perimeter = cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
                        
                        if len(approx) == 4:
                            area = cv2.contourArea(contour)
                            image_area = image.shape[0] * image.shape[1]
                            
                            if area > image_area * 0.03 and area < image_area * 0.95:
                                corners = approx.reshape(4, 2).tolist()
                                
                                if self._is_valid_rectangle(corners):
                                    confidence = min(0.5, self._calculate_confidence(corners, area, image_area))
                                    rotation_angle = self._calculate_rotation(corners)
                                    
                                    debug_info['edge_only_success'] = True
                                    debug_info['edge_low'] = low
                                    debug_info['edge_high'] = high
                                    
                                    return {
                                        'detected': True,
                                        'contours': contour,
                                        'corners': corners,
                                        'rotation_angle': rotation_angle,
                                        'confidence': confidence
                                    }
            
            debug_info['edge_only_success'] = False
            return {'detected': False}
            
        except Exception as e:
            debug_info['edge_only_error'] = str(e)
            return {'detected': False}
    
    def _is_valid_rectangle(self, corners: List[Tuple[int, int]]) -> bool:
        """Check if the four points form a valid rectangle."""
        # Check for duplicate points
        unique_corners = [tuple(c) for c in corners]
        if len(set(unique_corners)) < 4:
            return False
        
        # Verify it is a convex polygon
        if not cv2.isContourConvex(np.array(corners, dtype=np.int32)):
            return False
        
        # Calculate distances
        distances = []
        for i in range(4):
            p1 = np.array(corners[i])
            p2 = np.array(corners[(i + 1) % 4])
            distance = np.linalg.norm(p2 - p1)
            distances.append(distance)
        
        # Check opposite sides
        ratio1 = distances[0] / distances[2] if distances[2] > 0 else 1
        ratio2 = distances[1] / distances[3] if distances[3] > 0 else 1
        
        # More lenient for documents
        return 0.3 < ratio1 < 3.0 and 0.3 < ratio2 < 3.0
    
    def _calculate_rotation(self, corners: List[Tuple[int, int]]) -> float:
        """Calculate the rotation angle of the document."""
        corners_sorted = self._sort_corners(corners)
        top_left, top_right = corners_sorted[0], corners_sorted[1]
        
        dx = top_right[0] - top_left[0]
        dy = top_right[1] - top_left[1]
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        return angle
    
    def _calculate_confidence(self, corners: List[Tuple[int, int]], area: float, image_area: float) -> float:
        """Calculate confidence score."""
        area_ratio = area / image_area
        area_confidence = min(1.0, area_ratio * 10) if area_ratio < 0.1 else 1.0
        
        corners_sorted = self._sort_corners(corners)
        
        # Check aspect ratio
        width1 = np.linalg.norm(np.array(corners_sorted[0]) - np.array(corners_sorted[1]))
        width2 = np.linalg.norm(np.array(corners_sorted[3]) - np.array(corners_sorted[2]))
        height1 = np.linalg.norm(np.array(corners_sorted[0]) - np.array(corners_sorted[3]))
        height2 = np.linalg.norm(np.array(corners_sorted[1]) - np.array(corners_sorted[2]))
        
        avg_width = (width1 + width2) / 2
        avg_height = (height1 + height2) / 2
        
        aspect_ratio = max(avg_width, avg_height) / min(avg_width, avg_height) if min(avg_width, avg_height) > 0 else 1
        aspect_confidence = 1.0 if 0.5 < aspect_ratio < 2.5 else 0.5
        
        # Corner angles
        angle_confidence = self._check_corner_angles(corners_sorted)
        
        confidence = (area_confidence * 0.4 + aspect_confidence * 0.3 + angle_confidence * 0.3)
        return min(1.0, confidence)
    
    def _check_corner_angles(self, corners: List[Tuple[int, int]]) -> float:
        """Check if corners form right angles."""
        angles = []
        for i in range(4):
            p1 = np.array(corners[i])
            p2 = np.array(corners[(i + 1) % 4])
            p3 = np.array(corners[(i + 2) % 4])
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angles.append(np.degrees(angle))
        
        avg_angle = np.mean(angles)
        angle_diff = abs(avg_angle - 90)
        
        if angle_diff < 15:
            return 1.0
        elif angle_diff < 30:
            return 0.7
        elif angle_diff < 45:
            return 0.4
        else:
            return 0.2
    
    def _sort_corners(self, corners: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Sort corners in clockwise order."""
        centroid = np.mean(corners, axis=0)
        
        def get_angle(point):
            return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
        
        corners_sorted = sorted(corners, key=get_angle)
        
        # Find top-left (smallest x+y)
        min_sum = float('inf')
        min_idx = 0
        for i, corner in enumerate(corners_sorted):
            corner_sum = corner[0] + corner[1]
            if corner_sum < min_sum:
                min_sum = corner_sum
                min_idx = i
        
        return corners_sorted[min_idx:] + corners_sorted[:min_idx]
    
    def correct_perspective(self, image: np.ndarray, corners: List[Tuple[int, int]]) -> np.ndarray:
        """Apply perspective correction."""
        corners_sorted = self._sort_corners(corners)
        
        # Calculate width and height
        width1 = np.linalg.norm(np.array(corners_sorted[0]) - np.array(corners_sorted[1]))
        width2 = np.linalg.norm(np.array(corners_sorted[3]) - np.array(corners_sorted[2]))
        height1 = np.linalg.norm(np.array(corners_sorted[0]) - np.array(corners_sorted[3]))
        height2 = np.linalg.norm(np.array(corners_sorted[1]) - np.array(corners_sorted[2]))
        
        max_width = max(int(width1), int(width2))
        max_height = max(int(height1), int(height2))
        
        target_width = max(max_width, 800)
        target_height = max(max_height, int(target_width * 1.3))
        
        if target_width > 2000:
            scale = 2000 / target_width
            target_width = 2000
            target_height = int(target_height * scale)
        
        dst_points = np.array([
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1]
        ], dtype=np.float32)
        
        src_points = np.array(corners_sorted, dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        corrected = cv2.warpPerspective(
            image, matrix, (target_width, target_height),
            flags=cv2.INTER_LANCZOS4
        )
        
        return corrected
    
    def visualize_detection(self, image: np.ndarray, detection_result: DocumentDetectionResult) -> np.ndarray:
        """Visualize detected boundaries and corners."""
        vis_image = image.copy()
        
        if detection_result.detected:
            if detection_result.contours is not None:
                cv2.drawContours(vis_image, [detection_result.contours], -1, (0, 255, 0), 3)
            
            if detection_result.corners:
                for i, corner in enumerate(detection_result.corners):
                    cv2.circle(vis_image, tuple(corner), 10, (0, 0, 255), -1)
                    cv2.putText(vis_image, f"{i+1}", 
                               (corner[0] + 10, corner[1] + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                center = tuple(np.mean(detection_result.corners, axis=0).astype(int))
                cv2.putText(vis_image, f"Angle: {detection_result.rotation_angle:.1f}°", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(vis_image, f"Confidence: {detection_result.confidence:.2f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis_image, "No document detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add debug info
        y_offset = 90
        for key, value in detection_result.debug_info.items():
            if key != 'contours' and not key.startswith('_'):
                cv2.putText(vis_image, f"{key}: {value}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y_offset += 25
        
        return vis_image