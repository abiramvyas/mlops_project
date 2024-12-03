# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r mlops_project/requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run the FastAPI service when the container launches
CMD ["uvicorn", "mlops_project.fastapi_service.main:app", "--host", "0.0.0.0", "--port", "80"]
