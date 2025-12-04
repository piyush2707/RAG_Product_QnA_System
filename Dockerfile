# --- 1. Use a Python base image ---
# We use a slim base image for smaller size and security
FROM python:3.11-slim

# --- 2. Set environment variables ---
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
# NOTE: MISTRAL_API_KEY must be passed at runtime, NOT hardcoded here.

# --- 3. Set the working directory ---
WORKDIR /app

# --- 4. Install dependencies ---
# Copy requirements and install dependencies first (faster build cache)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# --- 5. Copy the application code and data ---
# Copy application code
COPY app/ /app/app/
COPY scripts/ /app/scripts/
# Copy the built vector database (models/chroma_db)
COPY models/ /app/models/
# Copy the raw data (optional, but good for reproducibility)
COPY data/ /app/data/

# --- 6. Expose the port FastAPI runs on ---
EXPOSE 8000

# --- 7. Define the startup command ---
# Use Uvicorn to run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
