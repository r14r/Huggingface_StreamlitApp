import io
import os
import tempfile
from pathlib import Path
from typing import Iterable, List

import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import HfApi, InferenceClient
from PIL import Image

APP_TITLE = "Hugging Face Studio"

DEFAULT_TEXT_MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-9b-it",
]

DEFAULT_IMAGE_MODELS = [
    "black-forest-labs/FLUX.1-dev",
    "black-forest-labs/FLUX.1-schnell",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/sdxl-turbo",
]

DEFAULT_VIDEO_MODELS = [
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "zai-org/CogVideoX-2b",
    "ali-vilab/text-to-video-ms-1.7b",

    "calamansi/Wan2.2-TI2V-5B",
    "KlingTeam/ShotStream",
    "viberobin/Wan2.1-T2V-1.3B-VedioQuant",
    "dx8152/LTX2.3-Multifunctional",
    "Jovin12Dhas/Wan2.2-T2V-A14B",
    "the-sweater-cat/Wan2.1-Fun-V1.1-1.3B-Control-Diffusers",
    "TheStageAI/Elastic-Wan2.2-T2V-A14B-Diffusers",
    "zhendemeiyou/Wan2.2-TI2V-5B-Diffusers",
    "gajesh/LTX-2.3-mlx-q4",
    "gajesh/LTX-2.3-mlx-fp16",
    "Alibaba-DAMO-Academy/LumosX",
    "Kotajiro/LTX23-ruri_LoRA",
    "hlaaa/Open-Sora-v2",
    "2klpostive/wan-gguf",
    "Admmer/Wan2.2-TI2V-5B-Diffusers",
    "Calamdor/Wan2.2-T2V-A14B-BF16",
    "aaftabazad612/text-to-video-ms-1.7b",
    "H-EmbodVis/HyDRA",
    "snowytsai/Wan2.2-T2V-A14B",
    "snowytsai/Wan2.2-TI2V-5B-Diffusers",
    "snowytsai/Wan2.2-TI2V-5B",
    "snowytsai/Wan2.2-T2V-A14B-Diffusers",
]


load_dotenv(Path(__file__).with_name(".env"))
HF_TOKEN=os.getenv("HF_TOKEN", "")

st.set_page_config(page_title=APP_TITLE, page_icon="🤗", layout="wide")


@st.cache_resource(show_spinner=False)
def get_api(token: str | None) -> HfApi:
    return HfApi(token=token)


@st.cache_resource(show_spinner=False)
def get_client(token: str | None, provider: str | None) -> InferenceClient:
    kwargs = {"api_key": token} if token else {}
    if provider and provider != "auto":
        kwargs["provider"] = provider
    return InferenceClient(**kwargs)


@st.cache_data(ttl=900, show_spinner=False)
def search_models(token: str | None, search: str, task: str, limit: int = 20) -> List[str]:
    api = get_api(token)
    try:
        models = api.list_models(search=search or None, filter=task, sort="downloads", direction=-1, limit=limit)
        return [m.id for m in models]
    except Exception:
        return []


@st.cache_data(ttl=900, show_spinner=False)
def trending_models(token: str | None, task: str, limit: int = 20) -> List[str]:
    api = get_api(token)
    try:
        models = api.list_models(filter=task, sort="downloads", direction=-1, limit=limit)
        return [m.id for m in models]
    except Exception:
        return []



def save_binary_file(data: bytes, suffix: str) -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="hf_streamlit_"))
    path = tmp_dir / f"output{suffix}"
    path.write_bytes(data)
    return path



def coerce_text_response(response) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if hasattr(response, "choices"):
        parts: list[str] = []
        for choice in getattr(response, "choices", []) or []:
            message = getattr(choice, "message", None)
            content = getattr(message, "content", None)
            if content:
                parts.append(str(content))
        if parts:
            return "\n".join(parts)
    return str(response)



def stream_chat_completion(client: InferenceClient, model: str, messages: list[dict], max_tokens: int, temperature: float):
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    placeholder = st.empty()
    chunks: list[str] = []
    for chunk in stream:
        try:
            delta = chunk.choices[0].delta.content
        except Exception:
            delta = None
        if delta:
            chunks.append(delta)
            placeholder.markdown("".join(chunks))
    return "".join(chunks)



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


DEFAULT_VIDEO_PROMPT = "Ein Programmierer steht an ein em Whiteboard und erstelle eine Zeichnung"
def video_tab(token: str | None, provider: str | None):
    model = model_picker("Videogenerierung", token, "text-to-video", DEFAULT_VIDEO_MODELS, "video")
    prompt = st.text_area("Videoprompt", height=180, placeholder="Beschreibe das gewünschte Video.", value=DEFAULT_VIDEO_PROMPT)

    if st.button("Video generieren", type="primary", key="run_video"):
        if not prompt.strip():
            st.warning("Bitte einen Videoprompt eingeben.")
            return
        try:
            client = get_client(token, provider)
            with st.spinner("Video wird generiert ..."):
                video = client.text_to_video(prompt, model=model)
                if isinstance(video, (bytes, bytearray)):
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



def main() -> None:
    st.title(APP_TITLE)
    st.caption("Streamlit-App für Text-, Bild- und Videogenerierung über die Hugging Face API")

    token, provider = render_sidebar()

    tabs = st.tabs(["Text", "Bild", "Video"])
    with tabs[0]:
        text_tab(token, provider)
    with tabs[1]:
        image_tab(token, provider)
    with tabs[2]:
        video_tab(token, provider)


if __name__ == "__main__":
    main()
