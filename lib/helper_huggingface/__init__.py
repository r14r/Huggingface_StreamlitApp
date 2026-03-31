from typing import List

import streamlit as st
from huggingface_hub import HfApi, InferenceClient

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