# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# ----------------------------------------------------
# 1. INSTALL POPPLER (CRITICAL FIX FOR PDF PROCESSING)
# poppler-utils is the necessary package for PDF analysis
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    poppler-utils \
    # Clean up the package lists to keep the image small
    && rm -rf /var/lib/apt/lists/*
# ----------------------------------------------------

# Copy the current directory contents into the container at /app
COPY . /app

# Install any dependencies specified in requirements.txt
# Using pip directly here for simplicity, but you can swap to uv if preferred.
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the application using uvicorn (assuming you use FastAPI)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]