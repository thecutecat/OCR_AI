# Build script for Windows
param(
    [string]$ImageName = "ocr-ai",
    [string]$ImageTag = "latest",
    [switch]$Push
)

Write-Host "========================================" -ForegroundColor Green
Write-Host "Building OCR AI Docker Image" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# Build the image
Write-Host "`nBuilding Docker image..." -ForegroundColor Yellow
docker build -f docker/Dockerfile -t ${ImageName}:${ImageTag} .

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Image built successfully!" -ForegroundColor Green
    Write-Host "Image: ${ImageName}:${ImageTag}" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to build image" -ForegroundColor Red
    exit 1
}

# Show image info
Write-Host "`nImage Details:" -ForegroundColor Yellow
docker images ${ImageName}:${ImageTag}

# Optional: Push to registry
if ($Push) {
    Write-Host "`nPushing to Docker Hub..." -ForegroundColor Yellow
    docker push ${ImageName}:${ImageTag}
}

Write-Host "`nTo run with docker-compose:" -ForegroundColor Yellow
Write-Host "  docker-compose -f docker/docker-compose.yml up -d"