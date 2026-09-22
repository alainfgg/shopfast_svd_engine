# Shopfast — Motor de Recomendación SVD

Motor de recomendación colaborativa basado en **TruncatedSVD** con dos microservicios Flask orquestados por Docker.

## Arquitectura

| Servicio | Puerto | Función |
|---|---|---|
| **training-service** | 5001 | Entrena el modelo SVD a partir de eventos |
| **inference-service** | 5002 | Sirve recomendaciones (SVD o fallback popular) |

## Estructura

```
.
├── dataset/                  # Datos de eventos (events.csv)
├── training-service/
│   ├── app.py → train.py     # Servicio de entrenamiento (Flask)
│   ├── Dockerfile
│   └── requirements.txt
├── inference-service/
│   ├── app.py                # Servicio de inferencia (Flask)
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml
└── .gitignore
```

## Requerimientos

- Docker + Docker Compose
- Python 3.12 (solo para desarrollo local)

## Ejecución

```bash
# 1. Levantar ambos servicios
docker-compose up --build

# 2. Entrenar el modelo (en otra terminal)
curl.exe -X POST http://localhost:5001/train

# 3. Verificar estado del modelo
curl http://localhost:5001/health

# 4. Obtener recomendaciones
curl http://localhost:5002/recommendations

# 5. Recomendaciones para un usuario específico
curl http://localhost:5002/recommendations/1001
```

## Endpoints

**Training Service (5001):**
- `GET /` — Información del servicio
- `POST /train` — Entrena el modelo SVD
- `GET /health` — Estado del modelo

**Inference Service (5002):**
- `GET /` — Información del servicio
- `GET /recommendations` — Recomendaciones populares
- `GET /recommendations/<user_id>` — Recomendaciones para un usuario
- `GET /health` — Estado del modelo cargado

## Formato de respuesta

```json
{
  "user_id": "1001",
  "source": "popular_fallback",
  "latency_ms": 12.45,
  "recommendations": [
    {"product_id": "1821813", "score": 18545.0, "source": "popular"}
  ]
}
```
