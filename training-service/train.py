import json
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

app = Flask(__name__)
#enrutamiento y pesos de eventos
DATASET_PATH = os.getenv("DATASET_PATH", "/app/dataset/events.csv")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))
MODEL_PATH = MODEL_DIR / "model_svd.pkl"
TOP_SELLERS_PATH = MODEL_DIR / "top_sellers.json"
EVENT_WEIGHTS = {"view": 1.0, "cart": 2.5, "purchase": 5.0}
REQUIRED_COLUMNS = {"event_type", "product_id", "user_id"}


def train_model():
    events = pd.read_csv(DATASET_PATH)
    missing = REQUIRED_COLUMNS.difference(events.columns)
    # si faltan columnas requeridas, lanzamos error
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    events["weight"] = events["event_type"].map(EVENT_WEIGHTS).fillna(0.0)
    events = events[events["weight"] > 0].copy()
    events["user_id"] = events["user_id"].astype(str)
    events["product_id"] = events["product_id"].astype(str)

    user_ids = sorted(events["user_id"].unique())
    product_ids = sorted(events["product_id"].unique())
    user_index = {user_id: index for index, user_id in enumerate(user_ids)}
    product_index = {product_id: index for index, product_id in enumerate(product_ids)}
    grouped = (
        events.groupby(["user_id", "product_id"], as_index=False)["weight"]
        .sum()
    )
    user_item = csr_matrix(
        (
            grouped["weight"].to_numpy(),
            (
                grouped["user_id"].map(user_index).to_numpy(),
                grouped["product_id"].map(product_index).to_numpy(),
            ),
        ),
        shape=(len(user_ids), len(product_ids)),
    )
    #validamos que haya al menos dos usuarios y dos productos para entrenar nuestro modelo
    if user_item.shape[0] < 2 or user_item.shape[1] < 2:
        raise ValueError("At least two users and two products are required")

    components = max(1, min(20, min(user_item.shape) - 1))
    svd = TruncatedSVD(n_components=components, random_state=42)
    user_factors = svd.fit_transform(user_item)
    item_factors = svd.components_.T

    #MODELO SVD
    model = {
        "algorithm": "TruncatedSVD collaborative filtering",
        "user_ids": user_ids,
        "product_ids": product_ids,
        "user_item": user_item,
        "user_factors": user_factors,
        "item_factors": item_factors,
    }

    popularity = (
        events.groupby("product_id", as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "score"})
        .sort_values(["score", "product_id"], ascending=[False, True])
        .head(10)
    )
    top_sellers = [
        {"product_id": row.product_id, "score": round(float(row.score), 4)}
        for row in popularity.itertuples(index=False)
    ]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    TOP_SELLERS_PATH.write_text(json.dumps(top_sellers, indent=2), encoding="utf-8")
    return {
        "users": len(user_ids),
        "products": len(product_ids),
        "events": len(events),
        "model_path": str(MODEL_PATH),
        "top_sellers_path": str(TOP_SELLERS_PATH),
    }

#método para cargar el modelo entrenado
@app.route("/train", methods=["GET", "POST"])
def train():
    if request.method == "GET":
        return jsonify({"status": "use POST to train", "routes": ["/train", "/health"]})
    try:
        return jsonify({"status": "trained", **train_model()})
    except (OSError, ValueError, KeyError) as exc:
        return jsonify({"status": "error", "error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Training failed")
        return jsonify({"status": "error", "error": str(exc)}), 500
#metodo para verificar si el modelo ya fue cargado
@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_exists": MODEL_PATH.exists()})


@app.get("/")
def root():
    return jsonify({"status": "ok", "service": "training", "routes": ["/train", "/health"]})


@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "error", "error": f"Ruta no encontrada"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
