FROM python:3.11-slim

WORKDIR /app

#instalar compiladores y cabeceras de desarrollo de Python
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

#actualizar herramientas base antes de instalar dependencias
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

#instalar los requerimientos
RUN pip install --no-cache-dir -r requirements.txt

#training
COPY train.py .
RUN mkdir -p /app/models
EXPOSE 5001
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:5001/health || exit 1
CMD ["python", "train.py"]
