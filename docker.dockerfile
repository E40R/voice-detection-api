FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing
# (ffmpeg and libsndfile1 are REQUIRED for librosa)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Run the application
# We use 'sh -c' to read the PORT environment variable provided by Render/Railway
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]