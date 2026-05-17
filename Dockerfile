# SpaceX Falcon 9 Landing Prediction - Dockerfile

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8501

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p models data/raw data/processed logs output

# Expose ports
EXPOSE 8501 5000

# Default command
CMD ["streamlit", "run", "deployment/streamlit/app.py", "--server.port=8501", "--server.address=0.0.0.0"]