FROM python:3.11-slim

WORKDIR /app

# System deps for health check and matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend package into /app/backend/
COPY . /app/backend/

# Create directories for database persistence
RUN mkdir -p /app/db

# PYTHONPATH must point to /app so "from backend.x import y" resolves
ENV PYTHONPATH=/app
ENV TITANIC_CSV_PATH=/app/backend/data/titanic.csv
ENV DATABASE_URL=sqlite:////app/db/icberg.sqlite

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
