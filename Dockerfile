# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
# This includes main.py, workflow_with_memory.py, memory_manager.py, and requirements.txt
COPY . /app

# Install any dependencies specified in requirements.txt
# You MUST create a requirements.txt file listing all your Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the application using uvicorn (assuming you use FastAPI)
# Adjust main:app if your FastAPI instance is named differently in main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]