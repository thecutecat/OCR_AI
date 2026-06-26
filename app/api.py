from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
import json
import uvicorn
import sys
from datetime import datetime

from main import DocumentScannerApp

app = FastAPI(
    title="Smart Document Scanner & OCR Service",
    description="Upload document images for scanning and OCR processing",
    version="1.0.0"
)

# Add CORS middleware for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
config = {
    'document_type': 'business_card',
    'ocr_language': 'eng',
    'use_database': False,
    'tesseract_path': None  # Update this if needed
}

# Initialize scanner app
try:
    scanner_app = DocumentScannerApp(config)
    print("✓ Document Scanner initialized successfully")
except RuntimeError as e:
    print(f"✗ Error initializing OCR: {e}")
    print("Please install Tesseract or specify the path in config")
    sys.exit(1)

@app.get("/")
async def root():
    """Root endpoint with HTML interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smart Document Scanner & OCR</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #333; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
            .upload-area:hover { border-color: #666; }
            #result { margin-top: 20px; padding: 20px; background: #f5f5f5; border-radius: 5px; display: none; }
            #result pre { white-space: pre-wrap; word-wrap: break-word; }
            button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .success { background: #d4edda; color: #155724; }
            .error { background: #f8d7da; color: #721c24; }
            .loading { background: #fff3cd; color: #856404; }
            .endpoints { background: #e9ecef; padding: 15px; border-radius: 5px; margin: 10px 0; }
            .endpoints code { background: #fff; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>📄 Smart Document Scanner & OCR</h1>
        <p>Upload a document image for scanning and text extraction.</p>
        
        <div class="endpoints">
            <h3>API Endpoints:</h3>
            <ul>
                <li><code>POST /process</code> - Upload and process document</li>
                <li><code>GET /health</code> - Health check</li>
                <li><code>GET /docs</code> - Interactive API documentation</li>
            </ul>
        </div>
        
        <div class="upload-area">
            <h3>Upload Document</h3>
            <p>Supported formats: JPG, PNG, BMP, TIFF</p>
            <input type="file" id="fileInput" accept="image/*">
            <br><br>
            <button onclick="uploadFile()">Process Document</button>
        </div>
        
        <div id="status" class="status" style="display:none;"></div>
        <div id="result">
            <h3>Processing Result:</h3>
            <pre id="resultContent"></pre>
            <div id="imagePreview" style="margin-top: 20px;"></div>
        </div>
        
        <script>
            async function uploadFile() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                
                if (!file) {
                    alert('Please select a file first!');
                    return;
                }
                
                const status = document.getElementById('status');
                const result = document.getElementById('result');
                const resultContent = document.getElementById('resultContent');
                const imagePreview = document.getElementById('imagePreview');
                
                // Show loading status
                status.style.display = 'block';
                status.className = 'status loading';
                status.textContent = '⏳ Processing document... Please wait.';
                result.style.display = 'none';
                
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/process', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok && data.success) {
                        // Show success
                        status.className = 'status success';
                        status.textContent = '✅ Document processed successfully!';
                        
                        // Show result
                        result.style.display = 'block';
                        resultContent.textContent = JSON.stringify(data, null, 2);
                        
                        // Show extracted text
                        if (data.ocr_result && data.ocr_result.structured_data) {
                            const fields = data.ocr_result.structured_data;
                            let html = '<h4>Extracted Information:</h4><ul>';
                            for (const [key, value] of Object.entries(fields)) {
                                if (value) {
                                    html += `<li><strong>${key}:</strong> ${value}</li>`;
                                }
                            }
                            html += '</ul>';
                            imagePreview.innerHTML = html;
                        }
                        
                        // Show confidence
                        if (data.metadata && data.metadata.ocr_confidence) {
                            imagePreview.innerHTML += `<p><strong>Confidence:</strong> ${(data.metadata.ocr_confidence * 100).toFixed(1)}%</p>`;
                        }
                    } else {
                        // Show error
                        status.className = 'status error';
                        status.textContent = '❌ Error: ' + (data.error || 'Unknown error');
                        result.style.display = 'block';
                        resultContent.textContent = JSON.stringify(data, null, 2);
                    }
                } catch (error) {
                    status.className = 'status error';
                    status.textContent = '❌ Error: ' + error.message;
                }
            }
            
            // Auto-upload when file is selected
            document.getElementById('fileInput').addEventListener('change', function() {
                if (this.files.length > 0) {
                    uploadFile();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/process")
async def process_document(file: UploadFile = File(...)):
    """
    Process a document image for scanning and OCR.
    
    Args:
        file: Image file to process (JPG, PNG, BMP, TIFF)
    
    Returns:
        JSON with processing results including extracted text and metadata
    """
    try:
        # Validate file type
        allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        print(f"✓ Received file: {file.filename} ({len(content)} bytes)")
        
        # Process the image
        result = scanner_app.process_image(tmp_file_path)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        if not result.get('success', False):
            raise HTTPException(
                status_code=400, 
                detail=result.get('error', 'Processing failed')
            )
        
        # Add filename to result
        result['filename'] = file.filename
        
        print(f"✓ Processing complete for: {file.filename}")
        
        return JSONResponse(content=result)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"✗ Error processing {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "service": "Smart Document Scanner & OCR",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/status")
async def service_status():
    """Get service status and configuration."""
    return {
        "service": "Smart Document Scanner & OCR",
        "version": "1.0.0",
        "status": "running",
        "config": {
            "document_type": config.get('document_type'),
            "ocr_language": config.get('ocr_language'),
            "tesseract_path": config.get('tesseract_path')
        }
    }

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8090))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print("=" * 60)
    print("📄 Smart Document Scanner & OCR API Server")
    print("=" * 60)
    print(f"Server running on: http://{host}:{port}")
    print(f"API Documentation: http://{host}:{port}/docs")
    print(f"Web Interface: http://{host}:{port}/")
    print(f"Health Check: http://{host}:{port}/health")
    print("=" * 60)
    
    uvicorn.run(
        app, 
        host=host, 
        port=port,
        log_level="info"
    )