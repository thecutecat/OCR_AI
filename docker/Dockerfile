# Base image with Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables to avoid interactive prompts
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Jakarta \
    DEBIAN_FRONTEND=noninteractive

# Update package lists and install dependencies in separate steps
RUN apt-get update --fix-missing

# Install OpenCV dependencies (split into smaller groups)
RUN apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Tesseract OCR
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install build tools (optional)
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    g++ \
    cmake \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Build C++ module (optional)
WORKDIR /app/cpp_module
RUN python setup.py build_ext --inplace || echo "C++ module build skipped (optional)"

WORKDIR /app

# Create necessary directories
RUN mkdir -p outputs/debug outputs/processed outputs/json samples

# Expose ports
EXPOSE 8090
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8090/health || exit 1

# Default command
CMD ["python", "app/api.py"]