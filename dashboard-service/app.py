import os
import requests
import pandas as pd
import streamlit as st

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://inference-service:5002")
TRAINING_URL = os.getenv("TRAINING_URL", "http://training-service:5001")

st.set_page_config(page_title="ShopFast AI Recommender", layout="wide")
st.title("ShopFast — Motor de Recomendación SVD (PoC AWS)")
st.caption("Arquitectura Desacoplada | SLA Latencia < 200 ms | Meta Conversión: 1.5% → 3.5%")

# Barra lateral: Controles de simulación y MLOps
st.sidebar.header("Control de Simulación")
modo = st.sidebar.radio(
    "Escenario de Prueba:",
    ["Usuario Existente (SVD RAM)", "Usuario Nuevo (Cold Start / Fallback)", "Catálogo General (Home)"]
)

if modo == "Usuario Existente (SVD RAM)":
    user_id = st.sidebar.text_input("ID Usuario Anonimizado (HMAC)", value="1515915625519334445")
elif modo == "Usuario Nuevo (Cold Start / Fallback)":
    user_id = st.sidebar.text_input("ID Usuario sin Historial", value="nuevo_cliente_99999")
else:
    user_id = ""

top_n = st.sidebar.slider("Cantidad de productos (Top-N)", min_value=5, max_value=15, value=10)

st.sidebar.divider()
st.sidebar.subheader("Pipeline MLOps (Reentrenamiento)")
if st.sidebar.button("Ejecutar Entrenamiento Batch (POST /train)"):
    with st.spinner("Entrenando matriz TruncatedSVD..."):
        try:
            r_train = requests.post(f"{TRAINING_URL}/train", timeout=30)
            if r_train.status_code == 200:
                st.sidebar.success(f"Modelo actualizado: {r_train.json()}")
            else:
                st.sidebar.error(f"Error {r_train.status_code}: {r_train.text}")
        except Exception as exc:
            st.sidebar.error(f"No se pudo conectar a training-service: {exc}")

# Consulta en tiempo real a inference-service
endpoint = f"{INFERENCE_URL}/recommendations/{user_id}" if user_id else f"{INFERENCE_URL}/recommendations"
try:
    resp = requests.get(endpoint, params={"top_n": top_n}, timeout=5)
    data = resp.json()

    latencia = data.get("latency_ms", 0.0)
    origen = data.get("source", "desconocido")
    recs = data.get("recommendations", [])

    # KPIs de Arquitectura y Negocio
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latencia API", f"{latencia:.2f} ms", delta="Cumple SLA (<200 ms)" if latencia < 200 else "Excede SLA", delta_color="normal" if latencia < 200 else "inverse")
    col2.metric("Estrategia Activa", origen.upper())
    col3.metric("Ítems Retornados", len(recs))
    col4.metric("Impacto Proyectado", "3.5% Conv.", delta="+133% vs baseline (1.5%)")

    st.subheader("Resultados Personalizados para Decisión Comercial")
    if recs:
        df_recs = pd.DataFrame(recs)
        df_recs.index = df_recs.index + 1
        df_recs["afinidad_pct"] = (df_recs["score"].clip(lower=0) * 100).round(2).astype(str) + " %"
        st.dataframe(df_recs, use_container_width=True)
    else:
        st.warning("La API no devolvió productos.")

    with st.expander("Ver Contrato JSON Crudo (Evidencia REST API)"):
        st.json(data)

except Exception as e:
    st.error(f"Error conectando con inference-service ({endpoint}): {e}")