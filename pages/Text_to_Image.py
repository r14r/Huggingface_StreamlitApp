import io
import os
import tempfile
from pathlib import Path
from typing import Iterable, List

import streamlit as st

from huggingface_hub import HfApi, InferenceClient
from PIL import Image

from lib.helper_huggingface import get_client, search_models, trending_models
from lib.helper_streamlit import render_sidebar
from lib.config import APP_TITLE, DEFAULT_IMAGE_MODELS, DEFAULT_TEXT_MODELS

HF_TOKEN=os.getenv("HF_TOKEN", "")

st.set_page_config(page_title=APP_TITLE, page_icon="🤗", layout="wide")


token, provider = render_sidebar()

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

#
#
#
model = model_picker("Bildgenerierung", token, "text-to-image", DEFAULT_IMAGE_MODELS, "image")
prompt = st.text_area("Bildprompt", height=180, placeholder="Beschreibe das gewünschte Bild.")
negative_prompt = st.text_input("Negativer Prompt", placeholder="optional")

if st.button("Bild generieren", type="primary", key="run_image"):
    if not prompt.strip():
        st.warning("Bitte einen Bildprompt eingeben.")
    try:
        client = get_client(token, provider)
        with st.spinner("Bild wird generiert ..."):
            image = client.text_to_image(prompt, model=model, negative_prompt=negative_prompt or None)
            if isinstance(image, Image.Image):
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                data = buf.getvalue()
            else:
                raise TypeError(f"Unerwarteter Rückgabetyp: {type(image)}")
        st.image(data, caption=model, use_container_width=True)
        st.download_button("PNG herunterladen", data=data, file_name="generated.png", mime="image/png")
    except Exception as exc:
        st.error(f"Fehler bei der Bildgenerierung: {exc}")
