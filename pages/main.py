import streamlit as st

st.title("Hugging Face Studio")
st.write(
    "Hier kannst du verschiedene KI-Modelle ausprobieren, um Bilder, Videos und Texte zu generieren. "
    "Die Videoseite nutzt jetzt die v2-Logik mit HF-serverless, klarer Fehlerdiagnose und lokalem Fallback (Stub-Ausgabe)."
)

st.markdown(
    """
### Empfohlene Video-Strategie
- **hf-inference** nur für offiziell unterstützte HF-Modelle verwenden.
- **fal-ai** nur mit aktivierten Prepaid Credits verwenden.
- Bei HF-Fehlern wird automatisch auf **lokalen Fallback** umgeschaltet.
"""
)
