import cv2
import numpy as np
from typing import Tuple, Union
import logging

class ImageEnhancer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def enhance(self, image: np.ndarray, return_color: bool = True) -> np.ndarray:
        """
        Enhance image quality for better OCR readability.
        
        Args:
            image: Input image (BGR or grayscale)
            return_color: If True, return BGR image, else return grayscale
        
        Returns:
            Enhanced image (BGR if return_color=True, else grayscale)
        """
        # Store original shape info
        is_color = len(image.shape) == 3 and image.shape[2] == 3
        
        # Convert to grayscale if needed
        if is_color:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise using Non-local Means Denoising
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        
        # Apply adaptive thresholding
        thresholded = cv2.adaptiveThreshold(
            denoised, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 15, 2
        )
        
        # Morphological operations to clean up text
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Sharpen the image
        kernel_sharpen = np.array([[-1,-1,-1],
                                   [-1, 9,-1],
                                   [-1,-1,-1]])
        sharpened = cv2.filter2D(cleaned, -1, kernel_sharpen)
        
        # Return color image if requested
        if return_color:
            # Convert back to BGR (3-channel)
            return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        else:
            return sharpened
    
    def enhance_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image and return grayscale (for OCR).
        """
        return self.enhance(image, return_color=False)
    
    def enhance_color(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image and return color (for visualization).
        """
        return self.enhance(image, return_color=True)
    
    def contrast_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using histogram equalization.
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization
        enhanced = cv2.equalizeHist(image)
        
        # Return as color if input was color
        if len(image.shape) == 3:
            return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        return enhanced
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply advanced denoising techniques.
        """
        if len(image.shape) == 3:
            # Color image
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            # Grayscale image
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        
        return denoised
    
    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        Apply sharpening filter.
        """
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        
        sharpened = cv2.filter2D(image, -1, kernel)
        
        return sharpened