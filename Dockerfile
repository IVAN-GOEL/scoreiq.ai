FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer — only rebuilds if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

# Pre-train the RandomForest model during the Docker build.
# This means the model is ready the instant the container starts —
# no cold-start delay on first request.
# Generates data/model.pkl (~2 MB) from synthetic data.
RUN python credit_pipeline.py --train

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
