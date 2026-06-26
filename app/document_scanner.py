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
    
    def _ensure_color_image(self, image: np.ndarray) -> np.ndarray:
        """Ensure image is 3-channel BGR."""
        if len(image.shape) == 2:
            # Grayscale to BGR
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA to BGR
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # Already BGR
            return image
        else:
            # Unknown format, try to convert
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    def _ensure_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Ensure image is grayscale (1-channel)."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def detect_document(self, image: np.ndarray) -> DocumentDetectionResult:
        """
        Detect the primary document in the image using multiple strategies.
        """
        start_time = time.time()
        debug_info = {}
        
        # Ensure image is in the right format (BGR for processing)
        try:
            # Keep original for reference
            original = image.copy()
            
            # Ensure we have a color image for processing
            if len(image.shape) == 2:
                # Convert grayscale to BGR for processing
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                debug_info['input_format'] = 'grayscale_converted'
            elif len(image.shape) == 3 and image.shape[2] == 3:
                debug_info['input_format'] = 'bgr'
            else:
                debug_info['input_format'] = 'unknown'
                image = self._ensure_color_image(image)
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
        
        # Resize image for faster processing while maintaining aspect ratio
        height, width = image.shape[:2]
        scale = min(800 / width, 800 / height) if max(width, height) > 800 else 1.0
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Try multiple detection strategies
        strategies = [
            self._detect_via_edges,
            self._detect_via_threshold,
            self._detect_via_contour_approx
        ]
        
        result = {'detected': False}
        for strategy in strategies:
            if not result['detected']:
                result = strategy(image, debug_info)
        
        # Scale corners back to original image size if needed
        if result['detected'] and scale < 1.0:
            corners = result['corners']
            scaled_corners = [[int(c[0] / scale), int(c[1] / scale)] for c in corners]
            result['corners'] = scaled_corners
        
        processing_time = (time.time() - start_time) * 1000
        
        # If no document detected but we have some corners, try to validate
        if not result['detected'] and result.get('corners'):
            # Validate if corners form a reasonable rectangle
            if self._is_valid_rectangle(result['corners']):
                result['detected'] = True
                result['confidence'] = 0.3  # Low confidence but valid shape
        
        # Ensure confidence is between 0 and 1
        confidence = min(1.0, max(0.0, result.get('confidence', 0.0)))
        
        return DocumentDetectionResult(
            detected=result['detected'],
            contours=result.get('contours'),
            corners=result.get('corners'),
            rotation_angle=result.get('rotation_angle', 0.0),
            processing_time_ms=processing_time,
            confidence=confidence,
            debug_info=debug_info
        )
    
    def _detect_via_edges(self, image: np.ndarray, debug_info: Dict) -> Dict:
        """Detect document using edge detection."""
        try:
            # Ensure image is grayscale for edge detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Multiple edge detection strategies
            edge_variants = [
                cv2.Canny(blurred, 30, 100),
                cv2.Canny(blurred, 50, 150),
                cv2.Canny(blurred, 75, 200),
                cv2.Canny(blurred, 100, 250)
            ]
            
            for edge in edge_variants:
                # Dilate edges to close gaps
                kernel = np.ones((5, 5), np.uint8)
                dilated = cv2.dilate(edge, kernel, iterations=2)
                
                # Find contours
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                    continue
                
                # Sort contours by area
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                # Try to find a document contour
                for contour in contours[:10]:  # Check top 10 largest contours
                    # Approximate the contour to a polygon
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                    
                    # Check if the polygon has 4 corners (document)
                    if len(approx) == 4:
                        # Calculate the area of the contour
                        area = cv2.contourArea(contour)
                        image_area = image.shape[0] * image.shape[1]
                        
                        # Ensure the document occupies a significant portion of the image
                        if area > image_area * 0.05 and area < image_area * 0.95:
                            corners = approx.reshape(4, 2).tolist()
                            
                            # Calculate rotation angle
                            rotation_angle = self._calculate_rotation(corners)
                            
                            # Calculate confidence based on area ratio and corner angles
                            confidence = self._calculate_confidence(corners, area, image_area)
                            
                            debug_info['edge_method'] = 'success'
                            debug_info['edge_threshold'] = '50-150'
                            
                            return {
                                'detected': True,
                                'contours': contour,
                                'corners': corners,
                                'rotation_angle': rotation_angle,
                                'confidence': confidence
                            }
            
            debug_info['edge_method'] = 'failed'
            return {'detected': False}
            
        except Exception as e:
            debug_info['edge_method'] = f'error: {str(e)}'
            return {'detected': False}
    
    def _detect_via_threshold(self, image: np.ndarray, debug_info: Dict) -> Dict:
        """Detect document using adaptive thresholding."""
        try:
            # Ensure image is grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Morphological operations to clean up
            kernel = np.ones((5, 5), np.uint8)
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                debug_info['threshold_method'] = 'failed'
                return {'detected': False}
            
            # Sort contours by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            for contour in contours[:10]:
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                
                if len(approx) == 4:
                    area = cv2.contourArea(contour)
                    image_area = image.shape[0] * image.shape[1]
                    
                    if area > image_area * 0.05 and area < image_area * 0.95:
                        corners = approx.reshape(4, 2).tolist()
                        rotation_angle = self._calculate_rotation(corners)
                        confidence = self._calculate_confidence(corners, area, image_area)
                        
                        debug_info['threshold_method'] = 'success'
                        
                        return {
                            'detected': True,
                            'contours': contour,
                            'corners': corners,
                            'rotation_angle': rotation_angle,
                            'confidence': confidence
                        }
            
            debug_info['threshold_method'] = 'failed'
            return {'detected': False}
            
        except Exception as e:
            debug_info['threshold_method'] = f'error: {str(e)}'
            return {'detected': False}
    
    def _detect_via_contour_approx(self, image: np.ndarray, debug_info: Dict) -> Dict:
        """Detect document using contour approximation with different parameters."""
        try:
            # Ensure image is grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Try different blur and Canny parameters
            for blur_size in [(3, 3), (5, 5), (7, 7)]:
                blurred = cv2.GaussianBlur(gray, blur_size, 0)
                
                for canny_low, canny_high in [(30, 100), (50, 150), (75, 200), (100, 250)]:
                    edges = cv2.Canny(blurred, canny_low, canny_high)
                    
                    kernel = np.ones((3, 3), np.uint8)
                    dilated = cv2.dilate(edges, kernel, iterations=1)
                    
                    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if not contours:
                        continue
                    
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    
                    for contour in contours[:10]:
                        perimeter = cv2.arcLength(contour, True)
                        
                        # Try different epsilon values
                        for epsilon in [0.01, 0.015, 0.02, 0.025, 0.03]:
                            approx = cv2.approxPolyDP(contour, epsilon * perimeter, True)
                            
                            if len(approx) == 4:
                                area = cv2.contourArea(contour)
                                image_area = image.shape[0] * image.shape[1]
                                
                                if area > image_area * 0.03 and area < image_area * 0.95:
                                    corners = approx.reshape(4, 2).tolist()
                                    
                                    # Check if corners form a reasonable rectangle
                                    if self._is_valid_rectangle(corners):
                                        rotation_angle = self._calculate_rotation(corners)
                                        confidence = self._calculate_confidence(corners, area, image_area)
                                        
                                        debug_info['contour_method'] = 'success'
                                        debug_info['blur_size'] = blur_size
                                        debug_info['canny_threshold'] = f'{canny_low}-{canny_high}'
                                        debug_info['epsilon'] = epsilon
                                        
                                        return {
                                            'detected': True,
                                            'contours': contour,
                                            'corners': corners,
                                            'rotation_angle': rotation_angle,
                                            'confidence': confidence
                                        }
            
            debug_info['contour_method'] = 'failed'
            return {'detected': False}
            
        except Exception as e:
            debug_info['contour_method'] = f'error: {str(e)}'
            return {'detected': False}
    
    def _is_valid_rectangle(self, corners: List[Tuple[int, int]]) -> bool:
        """Check if the four points form a valid rectangle."""
        # Calculate distances between consecutive corners
        distances = []
        for i in range(4):
            p1 = np.array(corners[i])
            p2 = np.array(corners[(i + 1) % 4])
            distance = np.linalg.norm(p2 - p1)
            distances.append(distance)
        
        # Opposite sides should be approximately equal
        ratio1 = distances[0] / distances[2] if distances[2] > 0 else 1
        ratio2 = distances[1] / distances[3] if distances[3] > 0 else 1
        
        return 0.5 < ratio1 < 2.0 and 0.5 < ratio2 < 2.0
    
    def _calculate_rotation(self, corners: List[Tuple[int, int]]) -> float:
        """Calculate the rotation angle of the document."""
        # Sort corners
        corners_sorted = self._sort_corners(corners)
        top_left, top_right = corners_sorted[0], corners_sorted[1]
        
        # Calculate angle using arctan
        dx = top_right[0] - top_left[0]
        dy = top_right[1] - top_left[1]
        angle = np.arctan2(dy, dx) * 180 / np.pi
        
        return angle
    
    def _calculate_confidence(self, corners: List[Tuple[int, int]], area: float, image_area: float) -> float:
        """Calculate confidence score for document detection."""
        # Area ratio confidence
        area_ratio = area / image_area
        area_confidence = min(1.0, area_ratio * 10) if area_ratio < 0.1 else 1.0
        
        # Aspect ratio confidence (for 4 corners)
        corners_sorted = self._sort_corners(corners)
        
        # Calculate widths and heights
        width1 = np.linalg.norm(np.array(corners_sorted[0]) - np.array(corners_sorted[1]))
        width2 = np.linalg.norm(np.array(corners_sorted[3]) - np.array(corners_sorted[2]))
        height1 = np.linalg.norm(np.array(corners_sorted[0]) - np.array(corners_sorted[3]))
        height2 = np.linalg.norm(np.array(corners_sorted[1]) - np.array(corners_sorted[2]))
        
        avg_width = (width1 + width2) / 2
        avg_height = (height1 + height2) / 2
        
        # Check aspect ratio (typically between 0.5 and 2.0 for documents)
        aspect_ratio = max(avg_width, avg_height) / min(avg_width, avg_height) if min(avg_width, avg_height) > 0 else 1
        aspect_confidence = 1.0 if 0.7 < aspect_ratio < 1.8 else 0.5
        
        # Corner angle confidence
        angle_confidence = self._check_corner_angles(corners_sorted)
        
        # Combined confidence
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
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angles.append(np.degrees(angle))
        
        # Check if angles are close to 90 degrees
        avg_angle = np.mean(angles)
        angle_diff = abs(avg_angle - 90)
        
        if angle_diff < 10:
            return 1.0
        elif angle_diff < 20:
            return 0.7
        elif angle_diff < 30:
            return 0.4
        else:
            return 0.2
    
    def _sort_corners(self, corners: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Sort corners in clockwise order: top-left, top-right, bottom-right, bottom-left."""
        # Calculate the centroid
        centroid = np.mean(corners, axis=0)
        
        # Sort by angle from centroid
        def get_angle(point):
            angle = np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
            return angle
        
        corners_sorted = sorted(corners, key=get_angle)
        
        # Ensure top-left is first
        # Find the point with smallest x+y (top-left)
        min_sum = float('inf')
        min_idx = 0
        for i, corner in enumerate(corners_sorted):
            corner_sum = corner[0] + corner[1]
            if corner_sum < min_sum:
                min_sum = corner_sum
                min_idx = i
        
        # Rotate list so top-left is first
        corners_sorted = corners_sorted[min_idx:] + corners_sorted[:min_idx]
        
        return corners_sorted
    
    def correct_perspective(self, image: np.ndarray, corners: List[Tuple[int, int]]) -> np.ndarray:
        """Apply perspective correction to get a top-down view."""
        # Sort corners
        corners_sorted = self._sort_corners(corners)
        
        # Calculate the width and height for the corrected image
        width1 = np.linalg.norm(np.array(corners_sorted[0]) - np.array(corners_sorted[1]))
        width2 = np.linalg.norm(np.array(corners_sorted[3]) - np.array(corners_sorted[2]))
        height1 = np.linalg.norm(np.array(corners_sorted[0]) - np.array(corners_sorted[3]))
        height2 = np.linalg.norm(np.array(corners_sorted[1]) - np.array(corners_sorted[2]))
        
        # Use maximum dimensions for better quality
        max_width = max(int(width1), int(width2))
        max_height = max(int(height1), int(height2))
        
        # Maintain aspect ratio with minimum size
        target_width = max(max_width, 800)
        target_height = max(max_height, int(target_width * 1.3))
        
        # If target is too large, scale down
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
        
        # Calculate the perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply the perspective transformation
        corrected = cv2.warpPerspective(
            image, matrix, (target_width, target_height),
            flags=cv2.INTER_LANCZOS4  # High-quality interpolation
        )
        
        return corrected
    
    def visualize_detection(self, image: np.ndarray, detection_result: DocumentDetectionResult) -> np.ndarray:
        """Visualize detected boundaries and corners."""
        vis_image = image.copy()
        
        if detection_result.detected:
            # Draw contours
            if detection_result.contours is not None:
                cv2.drawContours(vis_image, [detection_result.contours], -1, (0, 255, 0), 3)
            
            # Draw corners with labels
            if detection_result.corners:
                for i, corner in enumerate(detection_result.corners):
                    cv2.circle(vis_image, tuple(corner), 10, (0, 0, 255), -1)
                    cv2.putText(vis_image, f"{i+1}", 
                               (corner[0] + 10, corner[1] + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Draw rotation angle
                center = tuple(np.mean(detection_result.corners, axis=0).astype(int))
                cv2.putText(vis_image, f"Angle: {detection_result.rotation_angle:.1f}°", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw confidence
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