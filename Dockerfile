# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml .

# Install dependencies
RUN pip install --no-cache-dir .

# Copy source code
COPY src/ ./src/
COPY data/ ./data/

# Expose API port
EXPOSE 8000

# Default command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
