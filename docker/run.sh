#!/bin/bash

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Running OCR AI with Docker${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if image exists
if ! docker images ocr-ai:latest | grep -q "ocr-ai"; then
    echo -e "${YELLOW}Image not found. Building...${NC}"
    ./docker/build.sh
fi

# Run with docker-compose
echo -e "${GREEN}Starting containers...${NC}"
docker-compose -f docker/docker-compose.yml up -d

# Wait for services to start
echo -e "${YELLOW}Waiting for services to start...${NC}"
sleep 10

# Check status
echo -e "\n${GREEN}Container Status:${NC}"
docker ps

echo -e "\n${GREEN}Service URLs:${NC}"
echo -e "  API: http://localhost:8090"
echo -e "  API Docs: http://localhost:8090/docs"
echo -e "  Health: http://localhost:8090/health"

echo -e "\n${GREEN}To view logs:${NC}"
echo -e "  docker-compose -f docker/docker-compose.yml logs -f ocr-service"

echo -e "\n${GREEN}To stop:${NC}"
echo -e "  docker-compose -f docker/docker-compose.yml down"