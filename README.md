# Smart Document Scanner & OCR Service

A production-ready document scanning and OCR system built with Python, OpenCV, and C++ integration.

## Features

- 📄 Document detection from images
- 🔄 Perspective correction
- 🎨 Image enhancement
- 📝 OCR text extraction
- 📊 Structured data parsing
- 💾 SQL Server integration
- 🐳 Docker containerization
- 🔬 Comprehensive testing

## Quick Start

### Using Docker

```bash
# Build and run with Docker Compose
docker-compose -f docker/docker-compose.yml up --build

# Or build and run individual container
docker build -f docker/Dockerfile -t document-scanner .
docker run -p 8080:8080 -v $(pwd)/outputs:/app/outputs document-scanner