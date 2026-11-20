import json
import boto3
import joblib
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

BUCKET_NAME = "agep3"  # Cambiar por el nombre de tu bucket S3
DATASET_KEY = "events.csv"

s3 = boto3.client("s3")
s3.download_file(BUCKET_NAME, DATASET_KEY, "events.csv")

df = pd.read_csv("events.csv")
weights = {"view": 1.0, "cart": 2.5, "purchase": 5.0}
df["weight"] = df["event_type"].map(weights).fillna(0.0)
df = df[df["weight"] > 0].copy()
df["user_id"] = df["user_id"].astype(str)
df["product_id"] = df["product_id"].astype(str)

user_ids = sorted(df["user_id"].unique())
product_ids = sorted(df["product_id"].unique())
u_idx = {u: i for i, u in enumerate(user_ids)}
p_idx = {p: i for i, p in enumerate(product_ids)}

grouped = df.groupby(["user_id", "product_id"], as_index=False)["weight"].sum()
matrix = csr_matrix(
    (grouped["weight"], (grouped["user_id"].map(u_idx), grouped["product_id"].map(p_idx))),
    shape=(len(user_ids), len(product_ids)),
)

svd = TruncatedSVD(n_components=max(1, min(20, min(matrix.shape) - 1)), random_state=42)
model = {
    "algorithm": "TruncatedSVD (AWS SageMaker)",
    "user_ids": user_ids,
    "product_ids": product_ids,
    "user_item": matrix,
    "user_factors": svd.fit_transform(matrix),
    "item_factors": svd.components_.T,
}

joblib.dump(model, "model_svd.pkl")
s3.upload_file("model_svd.pkl", BUCKET_NAME, "models/model_svd.pkl")
print(f"Artefacto model_svd.pkl publicado en s3://{BUCKET_NAME}/models/model_svd.pkl")