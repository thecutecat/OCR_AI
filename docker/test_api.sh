#!/bin/bash

echo "Testing OCR AI API in Docker"

# Health check
echo -e "\n1. Health Check:"
curl -s http://localhost:8090/health | python -m json.tool

# Test with sample image
echo -e "\n2. Processing Test Image:"
curl -X POST http://localhost:8090/process \
  -F "file=@samples/business_card_normal.jpg" \
  -H "Accept: application/json" \
  | python -m json.tool

echo -e "\n3. Check API Documentation:"
echo "Open http://localhost:8090/docs in your browser"