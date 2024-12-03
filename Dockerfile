# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY ./mlops_project /app/mlops_project

# The port is now handled by an environment variable in Cloud Run
ENV PORT 8080
EXPOSE ${PORT}

# Run the FastAPI service when the container launches
# Use the PORT environment variable
CMD ["sh", "-c", "uvicorn mlops_project.fastapi_service.main:app --host 0.0.0.0 --port ${PORT}"]