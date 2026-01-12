# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1         PYTHONDONTWRITEBYTECODE=1         PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8501

# Streamlit must listen on 0.0.0.0 inside the container
CMD ["streamlit", "run", "pitch-glide-direction-threshold.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false", "--server.fileWatcherType=none"]
