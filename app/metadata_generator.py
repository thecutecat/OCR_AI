import json
import time
from typing import Dict, Any
from datetime import datetime
import uuid

class MetadataGenerator:
    def __init__(self):
        pass
    
    def generate_metadata(
        self,
        image: any,
        detection_result: any,
        ocr_result: Dict[str, any],
        processing_time: float
    ) -> Dict[str, Any]:
        """
        Generate comprehensive metadata for the processed document.
        """
        metadata = {
            'document_detected': detection_result.detected,
            'rotation_angle': detection_result.rotation_angle if detection_result.detected else 0.0,
            'processing_time_ms': processing_time,
            'ocr_confidence': ocr_result.get('confidence', 0.0),
            'image_width': image.shape[1] if hasattr(image, 'shape') else 0,
            'image_height': image.shape[0] if hasattr(image, 'shape') else 0,
            'fields': ocr_result.get('structured_data', {}),
            'timestamp': datetime.now().isoformat(),
            'document_id': str(uuid.uuid4()),
            'document_type': 'business_card'  # This should be configurable
        }
        
        return metadata
    
    def save_metadata(self, metadata: Dict[str, Any], filepath: str) -> None:
        """
        Save metadata to a JSON file.
        """
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)