import os
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import streamlit as st

try:
    from huggingface_hub import InferenceClient
except Exception:  # pragma: no cover
    InferenceClient = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

APP_DIR = Path(__file__).parent
OUTPUT_DIR = APP_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN", "")
DEFAULT_HF_VIDEO_MODELS = [
    "Lightricks/LTX-Video",
    "tencent/HunyuanVideo",
]
DEFAULT_LOCAL_VIDEO_MODELS = [
    "zai-org/CogVideoX-2b",
    "ali-vilab/text-to-video-ms-1.7b",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
]


@dataclass
class RunResult:
    ok: bool
    mode: str
    model_id: str
    output_path: Optional[str] = None
    provider: Optional[str] = None
    message: str = ""
    details: str = ""


class VideoGenerationError(Exception):
    pass


class UnsupportedProviderModelError(VideoGenerationError):
    pass


def detect_device() -> str:
    if torch is None:
        return "python-only"
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def hf_text_to_video_supported(model_id: str) -> bool:
    return model_id in DEFAULT_HF_VIDEO_MODELS


def build_filename(prefix: str, suffix: str) -> Path:
    import time
    return OUTPUT_DIR / f"{prefix}_{int(time.time())}{suffix}"


def save_dummy_video(prompt: str, model_id: str) -> str:
    # Placeholder artifact so the flow is testable end-to-end without local video model install.
    path = build_filename("video_stub", ".json")
    path.write_text(json.dumps({"prompt": prompt, "model_id": model_id, "note": "Stub output generated because local inference is not implemented in this lightweight v2."}, indent=2), encoding="utf-8")
    return str(path)


def run_hf_serverless(prompt: str, model_id: str, provider: str) -> RunResult:
    if InferenceClient is None:
        return RunResult(False, "hf-serverless", model_id, provider=provider, message="huggingface_hub ist nicht installiert.")
    if not HF_TOKEN:
        return RunResult(False, "hf-serverless", model_id, provider=provider, message="HF_TOKEN fehlt.")
    if provider == "fal-ai":
        return RunResult(
            False,
            "hf-serverless",
            model_id,
            provider=provider,
            message="fal-ai benötigt Prepaid Credits. Nutze hf-inference oder local.",
        )
    if provider != "hf-inference":
        return RunResult(False, "hf-serverless", model_id, provider=provider, message=f"Unbekannter Provider: {provider}")
    if not hf_text_to_video_supported(model_id):
        return RunResult(
            False,
            "hf-serverless",
            model_id,
            provider=provider,
            message="Dieses Modell ist nicht in der v2-Whitelist für HF serverless Video.",
        )

    try:
        client = InferenceClient(provider="hf-inference", api_key=HF_TOKEN)
        # We intentionally call the generic request layer to keep the example resilient across client versions.
        # Some environments may not support text_to_video yet; error details are surfaced verbatim.
        response = client.post(json={"inputs": prompt}, model=model_id)
        output_path = build_filename("hf_video", ".bin")
        output_path.write_bytes(response if isinstance(response, (bytes, bytearray)) else bytes(str(response), "utf-8"))
        return RunResult(True, "hf-serverless", model_id, output_path=str(output_path), provider=provider, message="HF request erfolgreich.")
    except Exception as e:
        return RunResult(
            False,
            "hf-serverless",
            model_id,
            provider=provider,
            message=f"HF request fehlgeschlagen: {type(e).__name__}: {e}",
            details=traceback.format_exc(),
        )


def run_local(prompt: str, model_id: str) -> RunResult:
    # Lightweight v2: local implementation is intentionally pluggable.
    # This keeps the app runnable without forcing large model installs.
    try:
        output_path = save_dummy_video(prompt, model_id)
        return RunResult(True, "local", model_id, output_path=output_path, message="Lokaler Fallback wurde ausgeführt (Stub-Ausgabe).")
    except Exception as e:
        return RunResult(False, "local", model_id, message=f"Lokaler Lauf fehlgeschlagen: {type(e).__name__}: {e}", details=traceback.format_exc())


def smart_generate(prompt: str, preferred_mode: str, hf_model_id: str, local_model_id: str, provider: str) -> RunResult:
    if preferred_mode == "HF serverless":
        result = run_hf_serverless(prompt, hf_model_id, provider)
        if result.ok:
            return result
        local_result = run_local(prompt, local_model_id)
        local_result.message = f"HF fehlgeschlagen ({result.message}). Danach local fallback. {local_result.message}"
        local_result.details = result.details or local_result.details
        return local_result
    return run_local(prompt, local_model_id)


st.set_page_config(page_title="Video Generator v2", layout="wide")
st.title("Video Generator v2")
st.caption("HF serverless + local fallback + bessere Fehlerdiagnose")

with st.sidebar:
    st.subheader("System")
    st.write(f"Device: **{detect_device()}**")
    st.write(f"HF token gesetzt: **{'ja' if bool(HF_TOKEN) else 'nein'}**")
    st.write("fal-ai ist absichtlich als kostenpflichtig markiert.")

left, right = st.columns([2, 1])
with left:
    prompt = st.text_area("Prompt", value="A cinematic robot dancing in neon rain, highly detailed, dynamic camera motion", height=140)
with right:
    preferred_mode = st.selectbox("Ausführungsmodus", ["HF serverless", "Local"])
    provider = st.selectbox("HF Provider", ["hf-inference", "fal-ai"])
    hf_model_id = st.selectbox("HF Modell", DEFAULT_HF_VIDEO_MODELS + ["dx8152/LTX2.3-Multifunctional"])
    local_model_id = st.selectbox("Lokales Modell", DEFAULT_LOCAL_VIDEO_MODELS)

with st.expander("Warum dx8152/LTX2.3-Multifunctional problematisch sein kann", expanded=False):
    st.write(
        "Dieses Repo ist oft nicht direkt als standardisierter HF-serverless Text-to-Video Endpoint nutzbar. "
        "Die App behandelt es deshalb nicht als freigeschaltetes Standard-HF-Modell."
    )

if st.button("Video generieren", type="primary"):
    with st.status("Generierung läuft...", expanded=True) as status:
        st.write("1. Eingaben validieren")
        st.write(f"2. Modus: {preferred_mode}")
        st.write(f"3. Provider: {provider}")
        result = smart_generate(prompt, preferred_mode, hf_model_id, local_model_id, provider)
        if result.ok:
            status.update(label="Fertig", state="complete")
        else:
            status.update(label="Fehlgeschlagen", state="error")

    if result.ok:
        st.success(result.message)
        st.json({
            "mode": result.mode,
            "provider": result.provider,
            "model_id": result.model_id,
            "output_path": result.output_path,
        })
        if result.output_path and result.output_path.endswith(".json"):
            st.code(Path(result.output_path).read_text(encoding="utf-8"), language="json")
        elif result.output_path:
            st.info(f"Binäre Ausgabe gespeichert unter: {result.output_path}")
    else:
        st.error(result.message)
        if result.details:
            with st.expander("Technische Details"):
                st.code(result.details)

st.markdown("---")
st.subheader("Empfohlene Strategie")
st.markdown(
    """
- **hf-inference** nur für offiziell unterstützte HF-Modelle verwenden.
- **fal-ai** nur mit aktivierten Prepaid Credits verwenden.
- Problematische oder große Modelle in **Local** oder auf eigenen Endpoints ausführen.
- Beim HF-Fehler automatisch auf **local fallback** umschalten.
"""
)
