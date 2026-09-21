from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt

from gaf_transform import subject_image
from model import DiscriminatorClassifier

CLASS_NAMES = ["Healthy", "Slight PD", "Mild PD", "Moderate PD"]


@st.cache_resource
def load_model(checkpoint):
    model = DiscriminatorClassifier()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    return model


st.set_page_config(page_title="STRiDE", layout="wide")
st.title("STRiDE")
st.caption("Gait behavior and Parkinson's disease severity assessment")

checkpoint = st.sidebar.text_input(
    "Model checkpoint",
    "results/stgan/discriminator_best.pt",
)

uploaded = st.file_uploader("Upload processed GRF CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ground Reaction Force")
        chart_cols = [c for c in ["LTotal", "RTotal"] if c in df.columns]
        if chart_cols:
            st.line_chart(df[chart_cols])

    with col2:
        st.subheader("PD Stage")
        if Path(checkpoint).exists():
            image = subject_image(df, method="gadf")
            x = torch.from_numpy(image).unsqueeze(0)
            model = load_model(checkpoint)
            with torch.no_grad():
                _, logits = model(x)
                probs = torch.softmax(logits, dim=1).squeeze().numpy()

            result = pd.DataFrame({
                "Stage": CLASS_NAMES,
                "Probability": probs,
            })
            st.bar_chart(result.set_index("Stage"))
            st.metric("Prediction", CLASS_NAMES[int(np.argmax(probs))])
        else:
            st.info("Train the model first or select an existing checkpoint.")

    sensor_cols = [c for c in df.columns if c.startswith("L") and c[1:].isdigit()]
    if sensor_cols:
        st.subheader("Foot Pressure")
        pressure = df[sensor_cols].mean().to_numpy()
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.imshow(pressure.reshape(1, -1), aspect="auto")
        ax.set_yticks([])
        ax.set_xticks(range(len(sensor_cols)))
        ax.set_xticklabels(sensor_cols)
        st.pyplot(fig)
