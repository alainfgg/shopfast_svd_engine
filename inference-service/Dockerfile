FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

#actualizar herramientas base antes de instalar dependencias
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

#instalar los requerimientos
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
RUN mkdir -p /app/models
EXPOSE 5002
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:5002/health || exit 1
CMD ["python", "app.py"]
