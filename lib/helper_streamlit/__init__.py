import os
import streamlit as st

from lib.helper_huggingface import search_models, trending_models

def render_sidebar() -> tuple[str | None, str | None]:
    st.sidebar.title("Einstellungen")
    token = st.sidebar.text_input(
        "HF Token",
        value=os.getenv("HF_TOKEN", ""),
        type="password",
        help="Nutze einen Hugging Face User Access Token. Ohne Token sind viele Modelle oder Provider nicht nutzbar.",
    ) or None

    provider = st.sidebar.selectbox(
        "Inference Provider",
        ["auto", "hf-inference", "fal-ai", "replicate", "sambanova", "together"],
        index=0,
        help="Nicht jeder Provider unterstützt jede Aufgabe oder jedes Modell.",
    )

    st.sidebar.markdown(
        """
**Hinweise**
- Text nutzt Chat Completions.
- Bild nutzt `text_to_image`.
- Video nutzt `text_to_video`.
- Modellverfügbarkeit hängt vom Hub und vom gewählten Provider ab.
        """
    )
    return token, provider


def model_picker(title: str, token: str | None, task: str, defaults: list[str], key_prefix: str) -> str:
    st.subheader(title)
    col1, col2 = st.columns([2, 1])
    with col1:
        query = st.text_input("Modelsuche", key=f"{key_prefix}_query", placeholder="z. B. Qwen, FLUX, Wan")
    with col2:
        refresh = st.button("Liste aktualisieren", key=f"{key_prefix}_refresh")

    options = defaults.copy()
    if query or refresh:
        found = search_models(token, query, task)
        if found:
            options = found
    elif not query:
        found = trending_models(token, task)
        if found:
            options = found[:20]

    custom = st.text_input("Oder eigene Model-ID", key=f"{key_prefix}_custom", placeholder="owner/model")
    selected = st.selectbox("Modell", options=options, key=f"{key_prefix}_select") if options else ""
    return custom.strip() or selected
