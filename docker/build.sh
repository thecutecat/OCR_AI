#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Building OCR AI Docker Image${NC}"
echo -e "${GREEN}========================================${NC}"

# Set image name
IMAGE_NAME="ocr-ai"
IMAGE_TAG="latest"

# Build the image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -f docker/Dockerfile -t ${IMAGE_NAME}:${IMAGE_TAG} .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Image built successfully!${NC}"
    echo -e "${GREEN}Image: ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
else
    echo -e "${RED}✗ Failed to build image${NC}"
    exit 1
fi

# Show image info
echo -e "\n${YELLOW}Image Details:${NC}"
docker images ${IMAGE_NAME}:${IMAGE_TAG}

# Optional: Run with docker-compose
echo -e "\n${YELLOW}To run with docker-compose:${NC}"
echo -e "  docker-compose -f docker/docker-compose.yml up -d"