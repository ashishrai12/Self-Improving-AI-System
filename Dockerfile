# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Create directory for logs and models if they don't exist
RUN mkdir -p experiments models/saved

# Set the environment variable to ensure python output is sent straight to terminal
ENV PYTHONUNBUFFERED=1

# Run the feedback loop by default
CMD ["python", "pipeline/feedback_loop.py"]
