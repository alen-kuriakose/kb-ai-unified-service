# Use official Python image as base
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv


# Copy requirements and main files
COPY requirements.txt requirements.txt
COPY main.py main.py
COPY competency_framework.json competency_framework.json


# Install dependencies using uv with system flag
RUN uv pip install --system -r requirements.txt


# Copy only the app folder
COPY app/ app/

# Expose the port the app runs on
EXPOSE 8000

ENV PYTHONUNBUFFERED=1

# Healthcheck to verify Redis connection
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
