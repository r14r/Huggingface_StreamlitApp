import io
import os

import streamlit as st
from PIL import Image

from lib import coerce_text_response
from lib.config import APP_TITLE, DEFAULT_IMAGE_MODELS, DEFAULT_TEXT_MODELS, DEFAULT_VIDEO_MODELS
from lib.helper_huggingface import (
    DEFAULT_HF_VIDEO_MODELS,
    detect_device,
    get_client,
    smart_generate,
    stream_chat_completion,
)
from lib.helper_streamlit import model_picker, render_sidebar

DEFAULT_VIDEO_PROMPT = "A cinematic robot dancing in neon rain, highly detailed, dynamic camera motion"

st.set_page_config(page_title=APP_TITLE, page_icon="🤗", layout="wide")


def text_tab(token: str | None, provider: str | None):
    model = model_picker("Textgenerierung", token, "text-generation", DEFAULT_TEXT_MODELS, "text")
    system_prompt = st.text_area("System Prompt", value="You are a precise and helpful assistant.", height=100)
    user_prompt = st.text_area("User Prompt", height=220, placeholder="Schreibe hier deinen Prompt.")
    col1, col2 = st.columns(2)
    max_tokens = col1.slider("Max Tokens", min_value=64, max_value=4096, value=1024, step=64)
    temperature = col2.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
    stream = st.checkbox("Streaming", value=True)

    if st.button("Prompt ausführen", type="primary", key="run_text"):
        if not user_prompt.strip():
            st.warning("Bitte einen Prompt eingeben.")
            return
        try:
            client = get_client(token, provider)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            with st.spinner("Text wird generiert ..."):
                if stream:
                    content = stream_chat_completion(client, model, messages, max_tokens, temperature)
                else:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    content = coerce_text_response(response)
            st.markdown("### Ausgabe")
            st.markdown(content)
            st.download_button("Als TXT herunterladen", data=content.encode("utf-8"), file_name="output.txt", mime="text/plain")
        except Exception as exc:
            st.error(f"Fehler bei der Textgenerierung: {exc}")


def image_tab(token: str | None, provider: str | None):
    model = model_picker("Bildgenerierung", token, "text-to-image", DEFAULT_IMAGE_MODELS, "image")
    prompt = st.text_area("Bildprompt", height=180, placeholder="Beschreibe das gewünschte Bild.")
    negative_prompt = st.text_input("Negativer Prompt", placeholder="optional")

    if st.button("Bild generieren", type="primary", key="run_image"):
        if not prompt.strip():
            st.warning("Bitte einen Bildprompt eingeben.")
            return
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


def video_tab(token: str | None, provider: str | None):
    st.subheader("Videogenerierung (v2 integriert)")

    left, right = st.columns([2, 1])
    with left:
        prompt = st.text_area("Videoprompt", height=180, value=DEFAULT_VIDEO_PROMPT)
    with right:
        preferred_mode = st.selectbox("Ausführungsmodus", ["HF serverless", "Local"])
        hf_model_id = st.selectbox("HF Modell", DEFAULT_HF_VIDEO_MODELS + ["dx8152/LTX2.3-Multifunctional"])
        local_model_id = st.selectbox("Lokales Modell", DEFAULT_VIDEO_MODELS)

    if st.button("Video generieren", type="primary", key="run_video"):
        if not prompt.strip():
            st.warning("Bitte einen Videoprompt eingeben.")
            return

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
                st.code(open(result.output_path, "r", encoding="utf-8").read(), language="json")
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


def main() -> None:
    st.title(APP_TITLE)
    st.caption("Streamlit-App für Text-, Bild- und Videogenerierung über die Hugging Face API")

    token, provider = render_sidebar()

    with st.sidebar:
        st.subheader("System")
        st.write(f"Device: **{detect_device()}**")
        st.write(f"HF token gesetzt: **{'ja' if bool(token or os.getenv('HF_TOKEN')) else 'nein'}**")
        st.write("fal-ai ist absichtlich als kostenpflichtig markiert.")

    tabs = st.tabs(["Text", "Bild", "Video"])
    with tabs[0]:
        text_tab(token, provider)
    with tabs[1]:
        image_tab(token, provider)
    with tabs[2]:
        video_tab(token, provider)


if __name__ == "__main__":
    main()
