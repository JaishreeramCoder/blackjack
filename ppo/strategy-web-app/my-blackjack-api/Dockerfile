# Use an official lightweight Python image.
FROM python:3.9-slim

# Set the working directory.
WORKDIR /app

# Copy the requirements file and install dependencies.
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy your application code.
COPY . .

# Expose the port provided by the $PORT environment variable.
EXPOSE 7860

# Start your FastAPI app using uvicorn.
# The $PORT environment variable is provided by Hugging Face Spaces.
CMD sh -c "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"

