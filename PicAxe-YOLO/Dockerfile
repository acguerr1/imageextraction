# Use the latest slim version of Python
FROM python:slim

# Prevent Python from writing .pyc files and turn off buffering for better logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libzbar0 \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Creates a non-root user with UID and grants permission to access the /app folder
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# Set entrypoint to automatically run the YOLO detection script
ENTRYPOINT ["python", "yolo_detection.py"]
