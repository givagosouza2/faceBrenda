# app.py
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

import mediapipe as mp

st.set_page_config(page_title="FaceMesh 468 + FDR", layout="wide")
st.title("🙂 MediaPipe FaceMesh (468) colorido por Significativo_FDR_0.050")

st.markdown(
    """
**Entrada 1 (resultados):** `.xlsx` ou `.csv` com colunas:
- `Marcador` (ex.: `ponto_0` ... `ponto_467`)
- `Significativo_FDR_0.050` (True/False)

**Entrada 2 (imagem):** uma foto/frame com 1 rosto visível.

**Cores:**
- False = **vermelho**
- True = **verde**
"""
)

# -------------------------
# Uploads
# -------------------------
colA, colB = st.columns(2)
with colA:
    res_file = st.file_uploader("1) Resultados (.xlsx ou .csv)", type=["xlsx", "csv"])
with colB:
    img_file = st.file_uploader("2) Imagem do rosto (.png/.jpg/.jpeg)", type=["png", "jpg", "jpeg"])

@st.cache_data(show_spinner=False)
def load_results(file) -> pd.DataFrame:
    if file is None:
        return None
    name = file.name.lower()
    if name.endswith(".xlsx"):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)
    return df

def extract_idx(marker) -> int:
    m = re.search(r"(\d+)", str(marker))
    if not m:
        return None
    return int(m.group(1))

df = load_results(res_file)

if df is None or img_file is None:
    st.info("Envie o arquivo de resultados e uma imagem para plotar o FaceMesh.")
    st.stop()

# -------------------------
# Validações do dataframe
# -------------------------
required_cols = ["Marcador", "Significativo_FDR_0.050"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Colunas ausentes: {missing}. Colunas encontradas: {list(df.columns)}")
    st.stop()

df = df.copy()
df["idx"] = df["Marcador"].apply(extract_idx)
if df["idx"].isna().any():
    bad = df.loc[df["idx"].isna(), "Marcador"].head(10).tolist()
    st.error(f"Não consegui extrair índice numérico de alguns 'Marcador'. Exemplos: {bad}")
    st.stop()

# Converte para booleano (caso venha como texto)
sig = df["Significativo_FDR_0.050"]
if sig.dtype != bool:
    df["Significativo_FDR_0.050"] = (
        sig.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    )

if df["Significativo_FDR_0.050"].isna().any():
    st.error("A coluna Significativo_FDR_0.050 não pôde ser convertida para True/False.")
    st.stop()

# Mapa idx -> True/False
sig_map = dict(zip(df["idx"].astype(int).values, df["Significativo_FDR_0.050"].values))

# -------------------------
# Controles
# -------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    max_faces = st.number_input("max_num_faces", 1, 5, 1)
with c2:
    min_det = st.slider("min_detection_confidence", 0.0, 1.0, 0.5, 0.05)
with c3:
    min_track = st.slider("min_tracking_confidence", 0.0, 1.0, 0.5, 0.05)
with c4:
    point_size = st.slider("Tamanho do ponto", 5, 200, 25)

draw_edges = st.checkbox("Desenhar conexões (tesselation)", value=True)
refine = st.checkbox("refine_landmarks (mais detalhado em olhos/lábios)", value=False)

# -------------------------
# Roda MediaPipe FaceMesh na imagem
# -------------------------
image = Image.open(img_file).convert("RGB")
img_np = np.array(image)
h, w = img_np.shape[:2]

mp_face_mesh = mp.solutions.face_mesh
# FACEMESH_TESSELATION é o conjunto de arestas usado para desenhar a malha
# (faz parte da solução FaceMesh).
tesselation = mp_face_mesh.FACEMESH_TESSELATION

with st.spinner("Rodando MediaPipe FaceMesh na imagem..."):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=int(max_faces),
        refine_landmarks=bool(refine),
        min_detection_confidence=float(min_det),
        min_tracking_confidence=float(min_track),
    ) as face_mesh:

        results = face_mesh.process(img_np)

if not results.multi_face_landmarks:
    st.error("Nenhum rosto detectado nessa imagem. Tente outra foto/frame mais nítido e frontal.")
    st.stop()

# Por simplicidade, plota o 1º rosto detectado
landmarks = results.multi_face_landmarks[0].landmark

# Garante que temos 468 (modelo clássico); o FaceMesh descreve 468 landmarks. :contentReference[oaicite:1]{index=1}
n = min(468, len(landmarks))

# Extrai coordenadas (x,y normalizadas -> pixels)
xs = np.array([landmarks[i].x * w for i in range(n)])
ys = np.array([landmarks[i].y * h for i in range(n)])

# Cores por significância (True=verde, False=vermelho; se não existir no mapa, assume False)
sig_all = np.array([bool(sig_map.get(i, False)) for i in range(n)])
colors = np.where(sig_all, "green", "red")

# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img_np)

# (Opcional) desenha conexões da malha
if draw_edges:
    for a, b in tesselation:
        if a < n and b < n:
            ax.plot([xs[a], xs[b]], [ys[a], ys[b]], linewidth=0.5, alpha=0.4)

ax.scatter(xs, ys, s=point_size, c=colors)

ax.set_title("FaceMesh (468) — Verde=True, Vermelho=False (Significativo_FDR_0.050)")
ax.axis("off")
st.pyplot(fig, clear_figure=True)

st.caption(f"Pontos plotados: {n} | True (verde): {int(sig_all.sum())} | False (vermelho): {int((~sig_all).sum())}")
