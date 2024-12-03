# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the entire project structure
COPY . /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# The port is handled by an environment variable in Cloud Run
ENV PORT 8080
EXPOSE ${PORT}

# Run the FastAPI service when the container launches
CMD ["sh", "-c", "uvicorn fastapi_service.main:app --host 0.0.0.0 --port ${PORT}"]