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
CMD ["python", "app.py"]