from pathlib import Path

import streamlit as st

from dotenv import load_dotenv
from huggingface_hub import HfApi

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
    "tencent/HunyuanVideo",

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

DEFAULT_VIDEO_PROMPT = "Ein Programmierer steht an ein em Whiteboard und erstelle eine Zeichnung"


def init():
    load_dotenv(Path(__file__).with_name(".env"))

@st.cache_resource(show_spinner=False)
def get_api(token: str | None) -> HfApi:
    return HfApi(token=token)

