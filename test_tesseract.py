import pytesseract
import cv2
import os
import sys

def test_tesseract_installation():
    """Test Tesseract installation and find its path."""
    print("Testing Tesseract Installation")
    print("=" * 50)
    
    # Check common installation paths
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    ]
    
    found_path = None
    for path in possible_paths:
        if os.path.exists(path):
            found_path = path
            print(f"✓ Tesseract found at: {path}")
            break
    
    if not found_path:
        print("✗ Tesseract not found in common locations")
        print("\nPlease install Tesseract from:")
        print("https://github.com/UB-Mannheim/tesseract/wiki")
        print("\nAfter installation, you can specify the path:")
        print('python app/cli.py --input samples/ss.jpg --tesseract-path "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"')
        return False
    
    # Set Tesseract path
    pytesseract.pytesseract.tesseract_cmd = found_path
    
    # Check version
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract version: {version}")
    except Exception as e:
        print(f"✗ Error getting version: {e}")
        return False
    
    # Test with a simple image
    try:
        # Create a simple test image
        import numpy as np
        test_image = np.zeros((100, 300), dtype=np.uint8)
        test_image[20:80, 50:250] = 255  # White rectangle
        test_image[30:70, 60:240] = 0  # Black rectangle (simulating text)
        
        # Run OCR
        text = pytesseract.image_to_string(test_image)
        print(f"✓ OCR test completed. Output: '{text.strip()}'")
        
        # Get available languages
        langs = pytesseract.get_languages()
        print(f"✓ Available languages: {', '.join(langs[:10])}...")
        
        print("\n✓ Tesseract is working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ OCR test failed: {e}")
        return False

def check_tesseract_in_path():
    """Check if Tesseract is in PATH."""
    import subprocess
    
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            print("✓ Tesseract found in PATH")
            return True
        else:
            print("✗ Tesseract in PATH but not responding")
            return False
    except:
        print("✗ Tesseract not in PATH")
        return False

if __name__ == "__main__":
    # Check PATH first
    print("1. Checking if Tesseract is in PATH...")
    in_path = check_tesseract_in_path()
    
    if not in_path:
        print("\n2. Searching for Tesseract in common locations...")
    
    # Run full test
    success = test_tesseract_installation()
    
    if not success:
        print("\n" + "=" * 50)
        print("TROUBLESHOOTING STEPS:")
        print("=" * 50)
        print("\n1. Download Tesseract installer:")
        print("   https://github.com/UB-Mannheim/tesseract/wiki")
        print("\n2. Install and remember the installation path")
        print("\n3. Add to PATH or specify path in command:")
        print('   python app/cli.py --input samples/ss.jpg --tesseract-path "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"')
        print("\n4. Or set TESSERACT_PATH environment variable")
        print("   set TESSERACT_PATH=C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
        print("\n5. Restart your terminal")