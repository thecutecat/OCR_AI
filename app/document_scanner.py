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

class DocumentScanner:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_document(self, image: np.ndarray) -> DocumentDetectionResult:
        """
        Detect the primary document in the image using multiple strategies.
        Falls back to text-based detection if no clear borders are found.
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
        
        result = {'detected': False}
        detection_methods = []
        
        # Strategy 1: Border-based detection (original)
        detection_methods.append(("border_detection", self._detect_via_borders))
        
        # Strategy 2: Text-based detection (new for borderless images)
        detection_methods.append(("text_detection", self._detect_via_text))
        
        # Strategy 3: Edge-based detection
        detection_methods.append(("edge_detection", self._detect_via_edges))
        
        # Strategy 4: Adaptive threshold detection
        detection_methods.append(("adaptive_detection", self._detect_via_threshold))
        
        # Try each method
        best_result = None
        best_confidence = 0.0
        best_method = ""
        
        for method_name, method_func in detection_methods:
            try:
                result = method_func(image, debug_info)
                if result.get('detected', False):
                    debug_info[f'{method_name}_success'] = True
                    debug_info[f'{method_name}_confidence'] = result.get('confidence', 0)
                    
                    # Keep the best result based on confidence
                    conf = result.get('confidence', 0)
                    if conf > best_confidence:
                        best_confidence = conf
                        best_result = result
                        best_method = method_name
                else:
                    debug_info[f'{method_name}_failed'] = True
            except Exception as e:
                debug_info[f'{method_name}_error'] = str(e)
                self.logger.warning(f"Method {method_name} failed: {e}")
        
        # If we have a result, use it
        if best_result:
            result = best_result
            debug_info['best_method'] = best_method
            debug_info['best_confidence'] = best_confidence
        
        # If no document detected but image has content, try to create a region
        if not result.get('detected', False):
            debug_info['fallback_used'] = True
            result = self._detect_content_region(image, debug_info)
        
        processing_time = (time.time() - start_time) * 1000
        
        if result.get('detected', False):
            confidence = min(1.0, max(0.0, result.get('confidence', 0.3)))
            
            return DocumentDetectionResult(
                detected=True,
                contours=result.get('contours'),
                corners=result.get('corners'),
                rotation_angle=result.get('rotation_angle', 0.0),
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
    
    def _detect_via_borders(self, image: np.ndarray, debug_info: Dict) -> Dict:
        """Original border-based detection."""
        try:
            # Resize for faster processing
            height, width = image.shape[:2]
            scale = min(800 / width, 800 / height) if max(width, height) > 800 else 1.0
            
            if scale < 1.0:
                new_width = int(width * scale)
                new_height = int(height * scale)
                resized = cv2.resize(image, (new_width, new_height))
            else:
                resized = image
            
            if len(resized.shape) == 3:
                gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            else:
                gray = resized
            
            # Try multiple edge detection parameters
            edge_params = [
                (30, 100), (50, 150), (75, 200), (100, 250),
                (20, 80), (40, 120), (60, 180)
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
                    
                    for epsilon in [0.01, 0.015, 0.02, 0.025, 0.03]:
                        approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
                        
                        if len(approx) == 4:
                            area = cv2.contourArea(contour)
                            image_area = resized.shape[0] * resized.shape[1]
                            
                            if area > image_area * 0.03 and area < image_area * 0.95:
                                corners = approx.reshape(4, 2).tolist()
                                
                                if self._is_valid_rectangle(corners):
                                    confidence = self._calculate_confidence(corners, area, image_area)
                                    rotation_angle = self._calculate_rotation(corners)
                                    
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
                                        'confidence': confidence
                                    }
            
            return {'detected': False}
            
        except Exception as e:
            debug_info['border_error'] = str(e)
            return {'detected': False}
    
    def _detect_via_text(self, image: np.ndarray, debug_info: Dict) -> Dict:
        """
        Detect document region based on text content (for borderless images).
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Find text regions using morphological operations
            # 1. Apply adaptive threshold to isolate text
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # 2. Dilate to connect text characters into blocks
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=3)
            
            # 3. Close gaps to form solid text regions
            closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # 4. Find contours of text regions
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                debug_info['text_no_contours'] = True
                return {'detected': False}
            
            # 5. Find the bounding box of all text
            all_points = np.vstack(contours)
            x, y, w, h = cv2.boundingRect(all_points)
            
            # 6. Add some padding around text
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            # 7. Create corners from bounding box
            corners = [
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h]
            ]
            
            # 8. Calculate confidence based on text coverage
            text_area = np.sum(thresh > 0) / 255
            region_area = w * h
            coverage = text_area / region_area if region_area > 0 else 0
            
            confidence = min(1.0, coverage * 3)  # Coverage up to 33% is good for text
            
            # 9. Check if the region is significant
            image_area = image.shape[0] * image.shape[1]
            region_ratio = region_area / image_area
            
            if region_ratio > 0.05 and coverage > 0.01:
                debug_info['text_detection_success'] = True
                debug_info['text_coverage'] = coverage
                debug_info['region_ratio'] = region_ratio
                
                # Create a contour from corners
                contour = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
                
                return {
                    'detected': True,
                    'contours': contour,
                    'corners': corners,
                    'rotation_angle': 0.0,
                    'confidence': confidence
                }
            
            return {'detected': False}
            
        except Exception as e:
            debug_info['text_error'] = str(e)
            return {'detected': False}
    
    def _detect_via_edges(self, image: np.ndarray, debug_info: Dict) -> Dict:
        """Edge-based detection as fallback."""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Try different edge detection parameters
            for low in [30, 50, 70]:
                for high in [100, 150, 200]:
                    edges = cv2.Canny(blurred, low, high)
                    
                    # Dilate to connect edges
                    kernel = np.ones((3, 3), np.uint8)
                    dilated = cv2.dilate(edges, kernel, iterations=2)
                    
                    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if not contours:
                        continue
                    
                    # Find the largest contour
                    largest = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest)
                    image_area = image.shape[0] * image.shape[1]
                    
                    if area > image_area * 0.05:
                        # Get bounding rectangle
                        x, y, w, h = cv2.boundingRect(largest)
                        
                        # Create corners
                        corners = [
                            [x, y],
                            [x + w, y],
                            [x + w, y + h],
                            [x, y + h]
                        ]
                        
                        confidence = min(1.0, area / (image_area * 0.5))
                        
                        if confidence > 0.2:
                            debug_info['edge_fallback_success'] = True
                            contour = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
                            
                            return {
                                'detected': True,
                                'contours': contour,
                                'corners': corners,
                                'rotation_angle': 0.0,
                                'confidence': confidence
                            }
            
            return {'detected': False}
            
        except Exception as e:
            debug_info['edge_fallback_error'] = str(e)
            return {'detected': False}
    
    def _detect_content_region(self, image: np.ndarray, debug_info: Dict) -> Dict:
        """Final fallback: detect any significant content region."""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate image statistics
            mean_val = np.mean(gray)
            std_val = np.std(gray)
            
            # If image has significant variation, it has content
            if std_val > 10:
                # Find the bounding box of non-background pixels
                # Use Otsu threshold to separate foreground from background
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Find foreground pixels
                fg_pixels = np.where(thresh > 0)
                
                if len(fg_pixels[0]) > 0:
                    y_min, y_max = np.min(fg_pixels[0]), np.max(fg_pixels[0])
                    x_min, x_max = np.min(fg_pixels[1]), np.max(fg_pixels[1])
                    
                    # Add padding
                    padding = 30
                    x_min = max(0, x_min - padding)
                    x_max = min(image.shape[1], x_max + padding)
                    y_min = max(0, y_min - padding)
                    y_max = min(image.shape[0], y_max + padding)
                    
                    # Create corners
                    corners = [
                        [x_min, y_min],
                        [x_max, y_min],
                        [x_max, y_max],
                        [x_min, y_max]
                    ]
                    
                    area = (x_max - x_min) * (y_max - y_min)
                    image_area = image.shape[0] * image.shape[1]
                    region_ratio = area / image_area
                    
                    if region_ratio > 0.05:
                        debug_info['content_fallback_success'] = True
                        debug_info['region_ratio'] = region_ratio
                        
                        contour = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
                        
                        return {
                            'detected': True,
                            'contours': contour,
                            'corners': corners,
                            'rotation_angle': 0.0,
                            'confidence': 0.3
                        }
            
            debug_info['content_fallback_failed'] = True
            return {'detected': False}
            
        except Exception as e:
            debug_info['content_fallback_error'] = str(e)
            return {'detected': False}
    
    def _detect_via_threshold(self, image: np.ndarray, debug_info: Dict) -> Dict:
        """Adaptive threshold detection."""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            for block_size in [11, 15, 21, 31]:
                thresh = cv2.adaptiveThreshold(
                    gray, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, block_size, 2
                )
                
                kernel = np.ones((3, 3), np.uint8)
                closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
                
                contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                    continue
                
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                for contour in contours[:10]:
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                    
                    if len(approx) >= 4:
                        area = cv2.contourArea(contour)
                        image_area = image.shape[0] * image.shape[1]
                        
                        if area > image_area * 0.03 and area < image_area * 0.95:
                            # Get bounding rectangle
                            x, y, w, h = cv2.boundingRect(contour)
                            corners = [
                                [x, y],
                                [x + w, y],
                                [x + w, y + h],
                                [x, y + h]
                            ]
                            
                            confidence = min(1.0, area / (image_area * 0.5))
                            
                            if confidence > 0.2:
                                debug_info['adaptive_success'] = True
                                return {
                                    'detected': True,
                                    'contours': contour,
                                    'corners': corners,
                                    'rotation_angle': 0.0,
                                    'confidence': confidence
                                }
            
            return {'detected': False}
            
        except Exception as e:
            debug_info['adaptive_error'] = str(e)
            return {'detected': False}
    
    def _is_valid_rectangle(self, corners: List[Tuple[int, int]]) -> bool:
        """Check if points form a valid rectangle."""
        if len(corners) != 4:
            return False
        
        # Check for duplicate points
        unique_corners = [tuple(c) for c in corners]
        if len(set(unique_corners)) < 4:
            return False
        
        # Calculate side lengths
        distances = []
        for i in range(4):
            p1 = np.array(corners[i])
            p2 = np.array(corners[(i + 1) % 4])
            distance = np.linalg.norm(p2 - p1)
            distances.append(distance)
        
        # Check opposite sides (more lenient for borderless)
        ratio1 = distances[0] / distances[2] if distances[2] > 0 else 1
        ratio2 = distances[1] / distances[3] if distances[3] > 0 else 1
        
        return 0.2 < ratio1 < 5.0 and 0.2 < ratio2 < 5.0
    
    def _calculate_rotation(self, corners: List[Tuple[int, int]]) -> float:
        """Calculate rotation angle."""
        corners_sorted = self._sort_corners(corners)
        if len(corners_sorted) >= 2:
            top_left, top_right = corners_sorted[0], corners_sorted[1]
            dx = top_right[0] - top_left[0]
            dy = top_right[1] - top_left[1]
            return np.arctan2(dy, dx) * 180 / np.pi
        return 0.0
    
    def _calculate_confidence(self, corners: List[Tuple[int, int]], area: float, image_area: float) -> float:
        """Calculate confidence score."""
        area_ratio = area / image_area
        area_confidence = min(1.0, area_ratio * 10) if area_ratio < 0.1 else 1.0
        
        corners_sorted = self._sort_corners(corners)
        
        # Aspect ratio
        if len(corners_sorted) >= 4:
            width1 = np.linalg.norm(np.array(corners_sorted[0]) - np.array(corners_sorted[1]))
            width2 = np.linalg.norm(np.array(corners_sorted[3]) - np.array(corners_sorted[2]))
            height1 = np.linalg.norm(np.array(corners_sorted[0]) - np.array(corners_sorted[3]))
            height2 = np.linalg.norm(np.array(corners_sorted[1]) - np.array(corners_sorted[2]))
            
            avg_width = (width1 + width2) / 2
            avg_height = (height1 + height2) / 2
            
            aspect_ratio = max(avg_width, avg_height) / min(avg_width, avg_height) if min(avg_width, avg_height) > 0 else 1
            aspect_confidence = 1.0 if 0.3 < aspect_ratio < 3.0 else 0.5
        else:
            aspect_confidence = 0.5
        
        angle_confidence = self._check_corner_angles(corners_sorted) if len(corners_sorted) >= 4 else 0.5
        
        confidence = (area_confidence * 0.4 + aspect_confidence * 0.3 + angle_confidence * 0.3)
        return min(1.0, max(0.0, confidence))
    
    def _check_corner_angles(self, corners: List[Tuple[int, int]]) -> float:
        """Check corner angles."""
        if len(corners) < 4:
            return 0.5
        
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
        
        if angle_diff < 20:
            return 1.0
        elif angle_diff < 40:
            return 0.7
        elif angle_diff < 60:
            return 0.4
        else:
            return 0.2
    
    def _sort_corners(self, corners: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Sort corners clockwise starting from top-left."""
        if not corners:
            return []
        
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
    
    def correct_perspective(self, image: np.ndarray, corners: List[Tuple[int, int]]) -> np.ndarray:
        """Apply perspective correction."""
        if not corners or len(corners) < 4:
            return image
        
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