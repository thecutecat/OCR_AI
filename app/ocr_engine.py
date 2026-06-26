import pytesseract
import cv2
import numpy as np
from typing import Dict, List, Optional
import re
import json
import logging
import os
import subprocess
import sys

class OCREngine:
    def __init__(self, language='eng', document_type='business_card', tesseract_path=None):
        self.language = language
        self.document_type = document_type
        self.logger = logging.getLogger(__name__)
        
        # Set Tesseract path if provided
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            # Try to find Tesseract in common locations
            self._auto_detect_tesseract()
        
        # Verify Tesseract is available
        self._verify_tesseract()
        
        # Configure Tesseract
        self.custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz@.-+_/ '
    
    def _auto_detect_tesseract(self):
        """Auto-detect Tesseract installation path."""
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME')),
            '/usr/bin/tesseract',  # Linux
            '/usr/local/bin/tesseract',  # macOS
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                self.logger.info(f"Tesseract found at: {path}")
                return
        
        # Check if tesseract is in PATH
        try:
            result = subprocess.run(['tesseract', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info("Tesseract found in PATH")
                return
        except:
            pass
        
        self.logger.warning("Tesseract not found in common locations")
    
    def _verify_tesseract(self):
        """Verify Tesseract is working."""
        try:
            # Try to get version
            version = pytesseract.get_tesseract_version()
            self.logger.info(f"Tesseract version: {version}")
        except Exception as e:
            self.logger.error(f"Tesseract verification failed: {e}")
            self.logger.error("Please install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
            raise RuntimeError(f"Tesseract is not installed or not in PATH. Error: {e}")
    
    def extract_text(self, image: np.ndarray) -> Dict[str, any]:
        """
        Extract text from the enhanced image and structure it based on document type.
        """
        try:
            # Perform OCR
            text = pytesseract.image_to_string(
                image, 
                lang=self.language, 
                config=self.custom_config
            )
            
            # Parse the text based on document type
            structured_data = self._parse_text(text)
            
            # Get confidence metrics
            confidence = self._calculate_confidence(image, text)
            
            return {
                'raw_text': text,
                'structured_data': structured_data,
                'confidence': confidence
            }
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            raise RuntimeError(f"OCR extraction failed: {e}")
    
    def _parse_text(self, text: str) -> Dict[str, any]:
        """
        Parse extracted text based on document type.
        """
        lines = text.strip().split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        
        if self.document_type == 'business_card':
            return self._parse_business_card(lines)
        elif self.document_type == 'id_card':
            return self._parse_id_card(lines)
        elif self.document_type == 'receipt':
            return self._parse_receipt(lines)
        else:
            return {'full_text': text}
    
    def _parse_business_card(self, lines: List[str]) -> Dict[str, str]:
        """
        Parse business card text into structured fields.
        """
        result = {
            'name': '',
            'company': '',
            'email': '',
            'phone': ''
        }
        
        # Email pattern
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        
        # Phone pattern (Indonesia format)
        phone_pattern = r'(\+62|62|0)8[1-9][0-9]{6,10}'
        
        for line in lines:
            # Check for email
            email_match = re.search(email_pattern, line)
            if email_match and not result['email']:
                result['email'] = email_match.group()
                continue
            
            # Check for phone
            phone_match = re.search(phone_pattern, line)
            if phone_match and not result['phone']:
                result['phone'] = phone_match.group()
                continue
            
            # Check for company (usually contains keywords)
            company_keywords = ['pt', 'ltd', 'inc', 'corp', 'co', 'perusahaan', 'company']
            if any(keyword in line.lower() for keyword in company_keywords):
                if not result['company']:
                    result['company'] = line
                    continue
            
            # Check for name (usually the first non-company, non-contact line)
            if not result['name'] and not result['company']:
                # Avoid common false positives
                if not any(keyword in line.lower() for keyword in ['email', 'phone', 'tel']):
                    result['name'] = line
        
        # Clean up results
        for key in result:
            result[key] = result[key].strip()
        
        return result
    
    def _parse_id_card(self, lines: List[str]) -> Dict[str, str]:
        """
        Parse ID card text into structured fields.
        """
        result = {
            'nik': '',
            'name': '',
            'birth_date': ''
        }
        
        # NIK pattern (Indonesia ID card - 16 digits)
        nik_pattern = r'\b[0-9]{16}\b'
        
        # Date pattern (multiple formats)
        date_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
            r'\b\d{2}/\d{2}/\d{4}\b',  # DD/MM/YYYY
            r'\b\d{2}-\d{2}-\d{4}\b'   # DD-MM-YYYY
        ]
        
        for line in lines:
            # Check for NIK
            nik_match = re.search(nik_pattern, line)
            if nik_match and not result['nik']:
                result['nik'] = nik_match.group()
                continue
            
            # Check for name
            if 'nama' in line.lower() or 'name' in line.lower():
                result['name'] = line.split(':', 1)[-1].strip() if ':' in line else line
                continue
            
            # Check for birth date
            for pattern in date_patterns:
                date_match = re.search(pattern, line)
                if date_match and not result['birth_date']:
                    result['birth_date'] = date_match.group()
                    break
        
        return result
    
    def _parse_receipt(self, lines: List[str]) -> Dict[str, any]:
        """
        Parse receipt text into structured fields.
        """
        result = {
            'merchant': '',
            'date': '',
            'total': 0.0,
            'items': []
        }
        
        # Date patterns
        date_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b\d{2}/\d{2}/\d{4}\b',
            r'\b\d{2}-\d{2}-\d{4}\b'
        ]
        
        # Total price patterns
        total_patterns = [
            r'total\s*[:\$]?\s*([0-9]+[,.]?[0-9]*)',
            r'subtotal\s*[:\$]?\s*([0-9]+[,.]?[0-9]*)',
            r'jumlah\s*[:\$]?\s*([0-9]+[,.]?[0-9]*)'
        ]
        
        for line in lines:
            # Check for merchant
            if not result['merchant'] and len(line) < 50:
                # Usually the first line or contains keywords
                merchant_keywords = ['store', 'shop', 'mart', 'restaurant', 'cafe']
                if any(keyword in line.lower() for keyword in merchant_keywords):
                    result['merchant'] = line
                elif not result['merchant'] and len(line) < 30:
                    result['merchant'] = line
            
            # Check for date
            for pattern in date_patterns:
                date_match = re.search(pattern, line)
                if date_match and not result['date']:
                    result['date'] = date_match.group()
                    break
            
            # Check for total
            for pattern in total_patterns:
                total_match = re.search(pattern, line, re.IGNORECASE)
                if total_match and result['total'] == 0:
                    total_str = total_match.group(1).replace(',', '.')
                    try:
                        result['total'] = float(total_str)
                    except ValueError:
                        pass
        
        return result
    
    def _calculate_confidence(self, image: np.ndarray, text: str) -> float:
        """
        Calculate OCR confidence based on various metrics.
        """
        try:
            # Get word-level confidence from Tesseract
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                return avg_confidence / 100.0  # Normalize to 0-1
        except:
            pass
        
        # Fallback: calculate confidence based on text length and quality
        text_length = len(text)
        if text_length == 0:
            return 0.0
        
        # Count alphanumeric characters
        alphanumeric = sum(1 for char in text if char.isalnum())
        ratio = alphanumeric / text_length
        
        # Basic quality metrics
        words = text.split()
        word_count = len(words)
        avg_word_length = text_length / max(word_count, 1)
        
        confidence = min(1.0, ratio * 0.7 + (avg_word_length / 10) * 0.3)
        return confidence