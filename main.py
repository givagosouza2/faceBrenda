# app.py
import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import mediapipe as mp

st.set_page_config(page_title="FaceMesh 468 + FDR", layout="wide")
st.title("🙂 MediaPipe FaceMesh colorido por Significativo_FDR_0.050")

st.markdown("""
Envie:
1. 📊 Arquivo com colunas:
   - `Marcador`
   - `Significativo_FDR_0.050`
2. 🖼️ Uma imagem contendo um rosto.

**Cores:**
- Verde = True
- Vermelho = False
""")

# -------------------------
# Uploads
# -------------------------
col1, col2 = st.columns(2)

with col1:
    result_file = st.file_uploader(
        "Upload resultados (.xlsx ou .csv)",
        type=["xlsx", "csv"]
    )

with col2:
    image_file = st.file_uploader(
        "Upload imagem (.png, .jpg, .jpeg)",
        type=["png", "jpg", "jpeg"]
    )

if result_file is None or image_file is None:
    st.info("Envie ambos os arquivos para continuar.")
    st.stop()

# -------------------------
# Carrega resultados
# -------------------------
def load_results(file):
    if file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    return pd.read_csv(file)

df = load_results(result_file)

required_cols = ["Marcador", "Significativo_FDR_0.050"]
if not all(col in df.columns for col in required_cols):
    st.error("Arquivo não contém as colunas necessárias.")
    st.stop()

def extract_idx(marker):
    match = re.search(r"(\d+)", str(marker))
    return int(match.group(1)) if match else None

df["idx"] = df["Marcador"].apply(extract_idx)

# Converte para booleano se necessário
if df["Significativo_FDR_0.050"].dtype != bool:
    df["Significativo_FDR_0.050"] = (
        df["Significativo_FDR_0.050"]
        .astype(str)
        .str.lower()
        .map({"true": True, "false": False})
    )

sig_map = dict(zip(df["idx"], df["Significativo_FDR_0.050"]))

# -------------------------
# Carrega imagem
# -------------------------
image = Image.open(image_file).convert("RGB")
image_np = np.array(image)
h, w = image_np.shape[:2]

# -------------------------
# MediaPipe FaceMesh
# -------------------------
mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5
) as face_mesh:

    results = face_mesh.process(image_np)

if not results.multi_face_landmarks:
    st.error("Nenhum rosto detectado na imagem.")
    st.stop()

landmarks = results.multi_face_landmarks[0].landmark

n_points = min(468, len(landmarks))

xs = np.array([landmarks[i].x * w for i in range(n_points)])
ys = np.array([landmarks[i].y * h for i in range(n_points)])

# Define cores
colors = [
    "green" if sig_map.get(i, False) else "red"
    for i in range(n_points)
]

# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(image_np)
ax.scatter(xs, ys, c=colors, s=25)
ax.axis("off")
ax.set_title("FaceMesh — Verde=True | Vermelho=False")

st.pyplot(fig)

st.caption(
    f"Total pontos: {n_points} | "
    f"Verde (True): {sum(sig_map.values())} | "
    f"Vermelho (False): {n_points - sum(sig_map.values())}"
)
