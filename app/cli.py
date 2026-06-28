import argparse
import os
import glob
from pathlib import Path
import json
from datetime import datetime
import cv2
import logging
import sys

from main import DocumentScannerApp

def main():
    parser = argparse.ArgumentParser(description='Smart Document Scanner & OCR Service')
    parser.add_argument('--input', required=True, help='Input image file or directory')
    parser.add_argument('--output', default='./outputs', help='Output directory')
    parser.add_argument('--document-type', default='auto', 
                       choices=['auto', 'business_card', 'id_card', 'receipt'],
                       help='Type of document to process (auto for automatic detection)')
    parser.add_argument('--ocr-language', default='eng', help='OCR language')
    parser.add_argument('--use-database', action='store_true', help='Save results to database')
    parser.add_argument('--db-connection', help='Database connection string')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--debug-dir', default='debug_output', help='Debug output directory')
    parser.add_argument('--tesseract-path', help='Path to tesseract.exe')
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        'document_type': args.document_type,  # 'auto' by default
        'ocr_language': args.ocr_language,
        'use_database': args.use_database,
        'db_connection_string': args.db_connection,
        'debug': args.debug,
        'debug_dir': args.debug_dir,
        'tesseract_path': args.tesseract_path
    }
    
    try:
        # Initialize application
        app = DocumentScannerApp(config)
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    
    # Process input
    if os.path.isfile(args.input):
        # Single file
        result = app.process_image(args.input)
        print(json.dumps(result, indent=2))
        
        # Show detected type
        if result.get('success'):
            print(f"\n📄 Detected Document Type: {result.get('document_type', 'unknown')}")
        
    elif os.path.isdir(args.input):
        # Directory - batch processing
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(args.input, f'*{ext}')))
        
        if not image_files:
            print(f"No image files found in {args.input}")
            return
        
        print(f"Found {len(image_files)} images to process")
        print(f"Document type: {args.document_type} (auto-detection if set to 'auto')")
        
        results = []
        success_count = 0
        fail_count = 0
        type_counts = {}
        
        for image_file in image_files:
            print(f"\nProcessing: {os.path.basename(image_file)}")
            result = app.process_image(image_file)
            results.append(result)
            
            if result.get('success', False):
                success_count += 1
                doc_type = result.get('document_type', 'unknown')
                type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
                print(f"  ✓ Success (Type: {doc_type})")
            else:
                fail_count += 1
                print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
        
        # Print summary
        print(f"\n{'='*50}")
        print(f"Batch Processing Summary:")
        print(f"  Total images: {len(image_files)}")
        print(f"  Successful: {success_count}")
        print(f"  Failed: {fail_count}")
        print(f"  Success rate: {success_count/len(image_files)*100:.1f}%")
        
        if type_counts:
            print(f"\n  Document Types Detected:")
            for doc_type, count in type_counts.items():
                print(f"    {doc_type}: {count}")
        
        # Save batch results
        output_dir = Path(args.output) / 'batch_results'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        batch_file = output_dir / f'batch_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(batch_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nBatch results saved to: {batch_file}")
    else:
        print(f"Error: Input path not found: {args.input}")

if __name__ == '__main__':
    main()