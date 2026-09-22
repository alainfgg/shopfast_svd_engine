import json
import os
import time
from pathlib import Path

import joblib
from flask import Flask, jsonify, request

app = Flask(__name__)
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/models/model_svd.pkl"))
TOP_SELLERS_PATH = Path(os.getenv("TOP_SELLERS_PATH", "/app/models/top_sellers.json"))
model = None


def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    return model


def load_top_sellers():
    if not TOP_SELLERS_PATH.exists():
        return []
    return json.loads(TOP_SELLERS_PATH.read_text(encoding="utf-8"))


def popular_recommendations(top_n):
    return [
        {"product_id": item["product_id"], "score": item["score"], "source": "popular"}
        for item in load_top_sellers()[:top_n]
    ]


@app.get("/health")
def health():
    loaded = model is not None or load_model() is not None
    return jsonify({"status": "ok", "model_loaded": loaded})


# apilamos las tres rutas posibles para que Flask las dirija a la misma función
@app.get("/recommendations/") #<- esta ruta es para cuando no se pasa un user_id
@app.get("/recommendations") 
@app.get("/recommendations/<user_id>") #<- con user_id
def recommendations(user_id=None):
    start = time.perf_counter() #contador latencia ms
    try:
        top_n = min(max(int(request.args.get("top_n", 5)), 1), 50)
    except ValueError:
        return jsonify({"error": "top_n must be an integer between 1 and 50"}), 400
    #mecanismo de resiliencia fallback: si no hay modelo entrenado, devolvemos recomendaciones populares
    loaded = model is not None or load_model() is not None
    if not loaded:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return jsonify({
            "user_id": user_id,
            "source": "resilience_cache",
            "latency_ms": latency_ms,
            "recommendations": popular_recommendations(top_n),
        })

    user_ids = model["user_ids"]
    product_ids = model["product_ids"]
    #
    if user_id is None or str(user_id) not in user_ids:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return jsonify({
            "user_id": user_id,
            "recommendations": popular_recommendations(top_n),
            "source": "general_popular" if user_id is None else "popular_fallback",
            "latency_ms": latency_ms,
        })

    user_index = user_ids.index(str(user_id))
    scores = model["user_factors"][user_index] @ model["item_factors"].T
    interacted = model["user_item"].getrow(user_index).toarray().ravel() > 0
    
    ranked_indices = sorted(
        range(len(product_ids)),
        key=lambda index: float(scores[index]),
        reverse=True,
    )
    
    recommendations = [
        {"product_id": product_ids[index], "score": round(float(scores[index]), 6), "source": "svd"}
        for index in ranked_indices
        if not interacted[index]
    ][:top_n]

    if len(recommendations) < top_n:
        seen = {item["product_id"] for item in recommendations}
        for item in popular_recommendations(top_n):
            if item["product_id"] not in seen:
                recommendations.append(item)
                seen.add(item["product_id"])
            if len(recommendations) == top_n:
                break
    #nuestra respuesta incluye la latencia en milisegundos
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return jsonify({
        "user_id": user_id,
        "source": "svd_with_popular_fallback" if len(recommendations) < top_n else "svd",
        "latency_ms": latency_ms,
        "recommendations": recommendations,
    })

@app.get("/")
def root():
    return jsonify({"status": "ok", "service": "inference", "routes": ["/health", "/recommendations"]})


@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "error", "error": f"Ruta no encontrada"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
