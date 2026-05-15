# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project into the container
COPY . .

# Create necessary directories
RUN mkdir -p results/plots logs policies mlruns

# Make startup script executable
RUN chmod +x start.sh

# Expose ports for Streamlit (8501), FastAPI (8000), and MLflow (5000)
EXPOSE 8501 8000 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Launch all services using the startup script
CMD ["./start.sh"]
