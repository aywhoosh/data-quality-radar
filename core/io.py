import io
import pandas as pd

def load_csv(uploaded_file, encoding_fallbacks=("utf-8", "latin-1")) -> pd.DataFrame:
    """
    Read a CSV from a Streamlit UploadedFile or a file path with simple robustness.
    """
    if hasattr(uploaded_file, "read"):
        raw = uploaded_file.read()
    else:
        with open(uploaded_file, "rb") as f:
            raw = f.read()

    last_err = None
    for enc in encoding_fallbacks:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception as e:
            last_err = e
            continue
    raise last_err
