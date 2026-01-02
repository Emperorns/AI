# Use a lightweight Python 2026-ready image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio/image processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Koyeb uses the PORT environment variable. We default to 8000.
ENV PORT=8000

# Start the bot
CMD ["python", "bot.py"]
