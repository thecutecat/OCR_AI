param(
    [string]$Service = "ocr-service",
    [switch]$Logs,
    [switch]$Stop
)

Write-Host "========================================" -ForegroundColor Green
Write-Host "Running OCR AI with Docker" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

if ($Stop) {
    Write-Host "Stopping containers..." -ForegroundColor Yellow
    docker-compose -f docker/docker-compose.yml down
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Containers stopped" -ForegroundColor Green
    }
    exit 0
}

# Check if image exists
$imageExists = docker images ocr-ai:latest -q
if (-not $imageExists) {
    Write-Host "Image not found. Building..." -ForegroundColor Yellow
    ./docker/build.ps1
}

# Run with docker-compose
Write-Host "Starting containers..." -ForegroundColor Yellow
docker-compose -f docker/docker-compose.yml up -d

# Wait for services
Write-Host "Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Show status
Write-Host "`nContainer Status:" -ForegroundColor Green
docker ps

Write-Host "`nService URLs:" -ForegroundColor Green
Write-Host "  API: http://localhost:8090" -ForegroundColor Cyan
Write-Host "  API Docs: http://localhost:8090/docs" -ForegroundColor Cyan
Write-Host "  Health: http://localhost:8090/health" -ForegroundColor Cyan

if ($Logs) {
    Write-Host "`nShowing logs..." -ForegroundColor Yellow
    docker-compose -f docker/docker-compose.yml logs -f $Service
}

Write-Host "`nTo stop containers:" -ForegroundColor Yellow
Write-Host "  ./docker/run.ps1 -Stop" -ForegroundColor Gray