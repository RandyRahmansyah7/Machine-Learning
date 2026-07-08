FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


# Copy application
COPY app.py .
COPY dashboard/ ./dashboard/

# Copy trained artifacts
COPY model_artifacts.joblib .
COPY inventory_dataset_clean.csv .


EXPOSE 8501


HEALTHCHECK CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" \
    || exit 1


ENTRYPOINT [
    "streamlit",
    "run",
    "app.py",
    "--server.port=8501",
    "--server.address=0.0.0.0"
]