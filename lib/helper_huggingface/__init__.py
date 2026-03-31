import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import streamlit as st
from huggingface_hub import HfApi, InferenceClient

from lib import save_binary_file

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

DEFAULT_HF_VIDEO_MODELS = [
    "Lightricks/LTX-Video",
    "tencent/HunyuanVideo",
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


def save_dummy_video(prompt: str, model_id: str) -> str:
    output_path = save_binary_file(
        json.dumps(
            {
                "prompt": prompt,
                "model_id": model_id,
                "note": "Stub output generated because local inference is not implemented in this app.",
            },
            indent=2,
        ).encode("utf-8"),
        ".json",
    )
    return str(output_path)


def run_hf_serverless(prompt: str, model_id: str, provider: str, token: str | None) -> RunResult:
    if not token:
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
            message="Dieses Modell ist nicht in der Whitelist für HF serverless Video.",
        )

    try:
        client = InferenceClient(provider="hf-inference", api_key=token)
        response = client.post(json={"inputs": prompt}, model=model_id)
        output_path = save_binary_file(
            response if isinstance(response, (bytes, bytearray)) else bytes(str(response), "utf-8"),
            ".bin",
        )
        return RunResult(True, "hf-serverless", model_id, output_path=str(output_path), provider=provider, message="HF request erfolgreich.")
    except Exception as exc:
        return RunResult(
            False,
            "hf-serverless",
            model_id,
            provider=provider,
            message=f"HF request fehlgeschlagen: {type(exc).__name__}: {exc}",
            details=traceback.format_exc(),
        )


def run_local(prompt: str, model_id: str) -> RunResult:
    try:
        output_path = save_dummy_video(prompt, model_id)
        return RunResult(True, "local", model_id, output_path=output_path, message="Lokaler Fallback wurde ausgeführt (Stub-Ausgabe).")
    except Exception as exc:
        return RunResult(False, "local", model_id, message=f"Lokaler Lauf fehlgeschlagen: {type(exc).__name__}: {exc}", details=traceback.format_exc())


def smart_generate(prompt: str, preferred_mode: str, hf_model_id: str, local_model_id: str, provider: str, token: str | None) -> RunResult:
    if preferred_mode == "HF serverless":
        result = run_hf_serverless(prompt, hf_model_id, provider, token)
        if result.ok:
            return result
        local_result = run_local(prompt, local_model_id)
        local_result.message = f"HF fehlgeschlagen ({result.message}). Danach local fallback. {local_result.message}"
        local_result.details = result.details or local_result.details
        return local_result
    return run_local(prompt, local_model_id)


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


@st.cache_resource(show_spinner=False)
def get_api(token: str | None) -> HfApi:
    return HfApi(token=token)


@st.cache_resource(show_spinner=False)
def get_client(api_key: str | None, provider: str | None) -> InferenceClient:
    kwargs = {"api_key": api_key} if api_key else {}

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

