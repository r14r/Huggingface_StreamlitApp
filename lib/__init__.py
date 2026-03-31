
from pathlib import Path
import tempfile

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