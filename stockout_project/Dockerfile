FROM python:3.10-slim

WORKDIR /app

# Install dependencies dulu agar layer di-cache saat kode berubah
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy sisa file project
COPY . .

# Latih model saat build (bisa dipindah ke entrypoint jika ingin retrain di runtime)
RUN python train_model.py

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]