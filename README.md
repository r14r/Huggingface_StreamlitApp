# Hugging Face Streamlit Studio

Eine Streamlit-App für:
- Textgenerierung
- Bildgenerierung
- Videogenerierung

Die App nutzt `huggingface_hub.InferenceClient` als einheitlichen Client für Inference auf dem Hugging Face Hub und über Inference Providers. Laut Hugging Face stellt `InferenceClient` eine einheitliche Schnittstelle für Inference API, Inference Endpoints und Inference Providers bereit. Für Streaming bei Chat Completions wird `stream=True` unterstützt. Für Bild- und Videogenerierung existieren eigene Task-Methoden wie `text_to_image` und `text_to_video`. citeturn148810search0turn148810search1turn148810search2turn148810search6turn148810search4

## Voraussetzungen

- Python 3.11+
- Ein Hugging Face User Access Token

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Dann `HF_TOKEN` in `.env` setzen.

## Start

```bash
streamlit run app.py
```

## Funktionen

### 1. Text
- Chat-ähnliche Ausführung über `client.chat.completions.create(...)`
- Optionales Streaming
- Frei wählbare Modell-ID

### 2. Bild
- Generierung über `client.text_to_image(...)`
- PNG-Download

### 3. Video
- Generierung über `client.text_to_video(...)`
- MP4-Download
- Funktioniert nur für Modelle/Provider, die Text-to-Video tatsächlich anbieten. Hugging Face dokumentiert Text-to-Video als eigene Task, aber die reale Verfügbarkeit hängt vom Provider und vom einzelnen Modell ab. citeturn148810search4turn148810search9turn148810search12

## Typische Modelle

### Text
- `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

### Bild
- `black-forest-labs/FLUX.1-Krea-dev`
- `stabilityai/stable-diffusion-xl-base-1.0`
- `black-forest-labs/FLUX.1-dev`

### Video
- `Wan-AI/Wan2.2-TI2V-5B`
- `Lightricks/LTX-Video`
- `tencent/HunyuanVideo`

Die Modellsuche in der App verwendet `HfApi.list_models(...)`, womit sich Modelle auf dem Hub suchen und filtern lassen. Hugging Face dokumentiert diese API als Python-Wrapper für Hub-Suchen und Repository-Metadaten. citeturn368688search0turn368688search4

## Hinweise

- Nicht jedes Modell ist serverless verfügbar.
- Nicht jeder Provider unterstützt jede Task.
- Für manche Modelle brauchst du Credits oder einen expliziten Provider.
- Falls ein Modell fehlschlägt, teste eine andere Modell-ID oder stelle den Provider auf `auto`.
