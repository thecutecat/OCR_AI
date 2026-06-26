# Base image with Python 3.14 (or latest)
FROM python:3.14-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libpq-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-ind \
    tesseract-ocr-jpn \
    && rm -rf /var/lib/apt/lists/*

# Install OpenCV and other Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Build C++ module
WORKDIR /app/cpp_module
RUN python setup.py build_ext --inplace

WORKDIR /app

# Create necessary directories
RUN mkdir -p outputs/debug outputs/processed outputs/json samples

# Expose API port
EXPOSE 8080

# Run the application
CMD ["python", "app/cli.py", "--input", "./samples", "--output", "./outputs"]