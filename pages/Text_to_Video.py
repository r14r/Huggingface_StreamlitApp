import io
import os
import tempfile
from pathlib import Path
from typing import Iterable, List

import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import HfApi
from PIL import Image

from lib import save_binary_file
from lib.config import APP_TITLE, DEFAULT_VIDEO_MODELS, DEFAULT_VIDEO_PROMPT
from lib.helper_huggingface import get_client
from lib.helper_streamlit import model_picker, render_sidebar

st.set_page_config(page_title=APP_TITLE, page_icon="🤗", layout="wide")

token, provider = render_sidebar()

model = model_picker("Videogenerierung", token, "text-to-video", DEFAULT_VIDEO_MODELS, "video")
prompt = st.text_area("Videoprompt", height=180, placeholder="Beschreibe das gewünschte Video.", value=DEFAULT_VIDEO_PROMPT)

if st.button("Video generieren", type="primary", key="run_video"):
    if not prompt.strip():
        st.warning("Bitte einen Videoprompt eingeben.")
    try:
        client = get_client(api_key=token, provider=provider)
        with st.spinner("Video wird generiert ..."):

                client = InferenceClient(
                    provider="fal-ai",
                    api_key=os.environ["HF_TOKEN"],
                )

                video = client.text_to_video(
                    "A young man walking on the street",
                    model="tencent/HunyuanVideo",
                )



            video = client.text_to_video(prompt, model=model)
            if isinstance(video, (bytes, bytearray, memoryview)):
                data = bytes(video)
            elif hasattr(video, "read"):
                data = video.read()
            else:
                raise TypeError(f"Unerwarteter Rückgabetyp: {type(video)}")
            path = save_binary_file(data, ".mp4")
        st.video(str(path))
        st.download_button("MP4 herunterladen", data=data, file_name="generated.mp4", mime="video/mp4")
    except Exception as exc:
        st.error(
            "Fehler bei der Videogenerierung: "
            f"{exc}. Prüfe Modell-ID, Provider-Unterstützung und ob dein Token für den gewählten Endpoint freigeschaltet ist."
        )


