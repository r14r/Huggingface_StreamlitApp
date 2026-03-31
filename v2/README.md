# Streamlit Video Generator v2

Diese v2 adressiert zwei konkrete Probleme:

1. **fal-ai ist kostenpflichtig** und benötigt Prepaid Credits.
2. **Nicht jedes HF-Repo ist automatisch serverless über den gewählten Provider nutzbar**.

## Enthaltene Verbesserungen

- Auswahl zwischen **HF serverless** und **Local**
- sichtbare Provider-Auswahl: `hf-inference` oder `fal-ai`
- klare Blockade für `fal-ai`, wenn keine Credits genutzt werden sollen
- besser lesbare Fehlermeldungen mit Exception-Typ
- automatischer **local fallback**, wenn HF fehlschlägt
- Systemanzeige: Token gesetzt, Device, Modellwahl
- problematisches Modell `dx8152/LTX2.3-Multifunctional` bewusst nicht als serverless-fähig behandelt

## Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=hf_xxx
streamlit run app.py
```

## Hinweise

- Die lokale Generierung ist in dieser kompakten v2 als **Stub-Fallback** umgesetzt, damit die App sofort lauffähig bleibt.
- Für echte lokale Video-Inferenz kannst du später Diffusers-Pipelines ergänzen, z. B. `CogVideoXPipeline` oder Wan-spezifische Pipelines.

## Warum dieser Umbau

Dein Fehler mit `fal-ai` war:

> Pre-paid credits are required to use provider fal-ai.

Das ist kein Modellfehler, sondern eine Kosten- und Provider-Sperre. Deshalb behandelt diese v2 `fal-ai` bewusst als kostenpflichtigen Sonderfall und lenkt den Flow auf `hf-inference` oder `local`.
