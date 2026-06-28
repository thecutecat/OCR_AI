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
    def __init__(self, language='eng', document_type='auto', tesseract_path=None):
        self.language = language
        self.document_type = document_type  # 'auto', 'business_card', 'id_card', 'receipt'
        self.logger = logging.getLogger(__name__)
        
        # Set Tesseract path if provided
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
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
            raise RuntimeError(f"Tesseract is not installed or not in PATH. Error: {e}")
    
    def _auto_detect_document_type(self, text: str) -> str:
        """
        Automatically detect document type based on text content.
        """
        text_lower = text.lower()
        
        # Check for ID Card patterns
        if re.search(r'\b[0-9]{16}\b', text):  # NIK pattern
            return 'id_card'
        if any(word in text_lower for word in ['nik', 'nama', 'tempat', 'tanggal lahir']):
            return 'id_card'
        
        # Check for Receipt patterns
        if any(word in text_lower for word in ['total', 'subtotal', 'payment', 'receipt', 'invoice', 
                                                'price', 'amount', 'cash', 'change', 'tax']):
            return 'receipt'
        if re.search(r'[0-9]+[,.]?[0-9]*\s*(total|subtotal|jumlah)', text_lower):
            return 'receipt'
        
        # Check for Business Card patterns
        if any(word in text_lower for word in ['email', '@', 'phone', 'tel', 'mobile', 
                                                'company', 'corp', 'inc', 'ltd']):
            return 'business_card'
        
        # Default
        return 'business_card'
    
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
            
            # Auto-detect document type if set to 'auto'
            detected_type = self.document_type
            if detected_type == 'auto':
                detected_type = self._auto_detect_document_type(text)
                self.logger.info(f"Auto-detected document type: {detected_type}")
            
            # Parse the text based on detected document type
            structured_data = self._parse_text(text, detected_type)
            
            # Get confidence metrics
            confidence = self._calculate_confidence(image, text)
            
            return {
                'raw_text': text,
                'structured_data': structured_data,
                'confidence': confidence,
                'detected_document_type': detected_type  # Return detected type
            }
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            raise RuntimeError(f"OCR extraction failed: {e}")
    
    def _parse_text(self, text: str, doc_type: str) -> Dict[str, any]:
        """
        Parse extracted text based on document type.
        """
        lines = text.strip().split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        
        if doc_type == 'business_card':
            return self._parse_business_card(lines)
        elif doc_type == 'id_card':
            return self._parse_id_card(lines)
        elif doc_type == 'receipt':
            return self._parse_receipt(lines)
        else:
            # Fallback: try to extract common fields
            return self._parse_generic(lines)
    
    def _parse_generic(self, lines: List[str]) -> Dict[str, str]:
        """
        Generic parser that tries to extract any meaningful fields.
        """
        result = {}
        
        # Try to find emails
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        for line in lines:
            email_match = re.search(email_pattern, line)
            if email_match:
                result['email'] = email_match.group()
                break
        
        # Try to find phone numbers
        phone_pattern = r'(\+62|62|0)8[1-9][0-9]{6,10}'
        for line in lines:
            phone_match = re.search(phone_pattern, line)
            if phone_match:
                result['phone'] = phone_match.group()
                break
        
        # Try to find names (lines with 2-3 words, not containing special chars)
        for line in lines:
            if len(line.split()) in [2, 3] and not re.search(r'[@+\d]', line):
                if 'name' not in result and len(line) < 30:
                    result['name'] = line
                    break
        
        # Add full text if no fields found
        if not result:
            result['full_text'] = '\n'.join(lines[:5])
        
        return result
    
    def _parse_business_card(self, lines: List[str]) -> Dict[str, str]:
        """Parse business card text into structured fields."""
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
            
            # Check for company
            company_keywords = ['pt', 'ltd', 'inc', 'corp', 'co', 'perusahaan', 'company']
            if any(keyword in line.lower() for keyword in company_keywords):
                if not result['company']:
                    result['company'] = line
                    continue
            
            # Check for name
            if not result['name'] and not result['company']:
                if not any(keyword in line.lower() for keyword in ['email', 'phone', 'tel']):
                    if len(line) < 40 and len(line.split()) <= 4:
                        result['name'] = line
        
        # Clean up results
        for key in result:
            result[key] = result[key].strip()
        
        return result
    
    def _parse_id_card(self, lines: List[str]) -> Dict[str, str]:
        """Parse ID card text into structured fields."""
        result = {
            'nik': '',
            'name': '',
            'birth_date': '',
            'birth_place': '',
            'gender': ''
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
                name_part = line.split(':', 1)[-1].strip() if ':' in line else line
                result['name'] = name_part
                continue
            
            # Check for birth place
            if 'tempat' in line.lower() or 'place' in line.lower():
                place_part = line.split(':', 1)[-1].strip() if ':' in line else line
                result['birth_place'] = place_part
                continue
            
            # Check for birth date
            for pattern in date_patterns:
                date_match = re.search(pattern, line)
                if date_match and not result['birth_date']:
                    result['birth_date'] = date_match.group()
                    break
            
            # Check for gender
            if 'gender' in line.lower() or 'jenis kelamin' in line.lower():
                gender_part = line.split(':', 1)[-1].strip() if ':' in line else line
                result['gender'] = gender_part
                continue
        
        return result
    
    def _parse_receipt(self, lines: List[str]) -> Dict[str, any]:
        """Parse receipt text into structured fields."""
        result = {
            'merchant': '',
            'date': '',
            'total': 0.0,
            'subtotal': 0.0,
            'tax': 0.0,
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
            r'jumlah\s*[:\$]?\s*([0-9]+[,.]?[0-9]*)',
            r'amount\s*[:\$]?\s*([0-9]+[,.]?[0-9]*)',
            r'grand total\s*[:\$]?\s*([0-9]+[,.]?[0-9]*)'
        ]
        
        # Item patterns (price and description)
        item_pattern = r'([a-zA-Z\s]+)\s+([0-9]+[,.]?[0-9]*)\s*$'
        
        for line in lines:
            # Check for merchant
            if not result['merchant'] and len(line) < 50:
                merchant_keywords = ['store', 'shop', 'mart', 'restaurant', 'cafe', 'market', 'supermarket']
                if any(keyword in line.lower() for keyword in merchant_keywords):
                    result['merchant'] = line
                elif not result['merchant'] and len(line) < 30 and len(line.split()) > 1:
                    result['merchant'] = line
            
            # Check for date
            for pattern in date_patterns:
                date_match = re.search(pattern, line)
                if date_match and not result['date']:
                    result['date'] = date_match.group()
                    break
            
            # Check for totals
            for pattern in total_patterns:
                total_match = re.search(pattern, line, re.IGNORECASE)
                if total_match:
                    total_str = total_match.group(1).replace(',', '.')
                    try:
                        value = float(total_str)
                        if 'subtotal' in pattern.lower():
                            result['subtotal'] = value
                        elif 'tax' in pattern.lower():
                            result['tax'] = value
                        else:
                            if result['total'] == 0:
                                result['total'] = value
                    except ValueError:
                        pass
            
            # Try to extract items
            item_match = re.match(item_pattern, line)
            if item_match and len(line) > 10:
                item_desc = item_match.group(1).strip()
                item_price = float(item_match.group(2).replace(',', '.'))
                result['items'].append({
                    'description': item_desc,
                    'price': item_price
                })
        
        return result
    
    def _calculate_confidence(self, image: np.ndarray, text: str) -> float:
        """Calculate OCR confidence based on various metrics."""
        try:
            # Get word-level confidence from Tesseract
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                return avg_confidence / 100.0
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