import os
from pathlib import Path

import streamlit as st

from lib.config import APP_TITLE, DEFAULT_VIDEO_MODELS
from lib.helper_huggingface import DEFAULT_HF_VIDEO_MODELS, detect_device, smart_generate
from lib.helper_streamlit import render_sidebar

DEFAULT_VIDEO_PROMPT = "A cinematic robot dancing in neon rain, highly detailed, dynamic camera motion"

st.set_page_config(page_title=APP_TITLE, page_icon="🤗", layout="wide")

st.title("Text zu Video")
st.caption("v2-Logik mit HF-serverless und lokalem Fallback")

token, provider = render_sidebar()

with st.sidebar:
    st.subheader("System")
    st.write(f"Device: **{detect_device()}**")
    st.write(f"HF token gesetzt: **{'ja' if bool(token or os.getenv('HF_TOKEN')) else 'nein'}**")

left, right = st.columns([2, 1])
with left:
    prompt = st.text_area("Videoprompt", height=180, placeholder="Beschreibe das gewünschte Video.", value=DEFAULT_VIDEO_PROMPT)
with right:
    preferred_mode = st.selectbox("Ausführungsmodus", ["HF serverless", "Local"])
    hf_model_id = st.selectbox("HF Modell", DEFAULT_HF_VIDEO_MODELS + ["dx8152/LTX2.3-Multifunctional"])
    local_model_id = st.selectbox("Lokales Modell", DEFAULT_VIDEO_MODELS)

if st.button("Video generieren", type="primary", key="run_video_page"):
    if not prompt.strip():
        st.warning("Bitte einen Videoprompt eingeben.")
    else:
        with st.status("Generierung läuft...", expanded=True) as status:
            st.write("1. Eingaben validieren")
            st.write(f"2. Modus: {preferred_mode}")
            st.write(f"3. Provider: {provider}")
            result = smart_generate(prompt, preferred_mode, hf_model_id, local_model_id, provider, token)
            status.update(label="Fertig" if result.ok else "Fehlgeschlagen", state="complete" if result.ok else "error")

        if result.ok:
            st.success(result.message)
            st.json(
                {
                    "mode": result.mode,
                    "provider": result.provider,
                    "model_id": result.model_id,
                    "output_path": result.output_path,
                }
            )
            if result.output_path and result.output_path.endswith(".json"):
                st.code(Path(result.output_path).read_text(encoding="utf-8"), language="json")
            elif result.output_path:
                st.info(f"Binäre Ausgabe gespeichert unter: {result.output_path}")
        else:
            st.error(result.message)
            if result.details:
                with st.expander("Technische Details"):
                    st.code(result.details)

with st.expander("Warum dx8152/LTX2.3-Multifunctional problematisch sein kann", expanded=False):
    st.write(
        "Dieses Repo ist oft nicht direkt als standardisierter HF-serverless Text-to-Video Endpoint nutzbar. "
        "Die App behandelt es deshalb nicht als freigeschaltetes Standard-HF-Modell."
    )
