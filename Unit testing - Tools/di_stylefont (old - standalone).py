# modules/di_stylefont_tester.py
"""
Azure Document Intelligence style detection tester (FR v3 first, DI v4 fallback)
-------------------------------------------------------------------------------
- Loads endpoint/key from repo-root .env (overrides OS env) or Streamlit secrets.
- Shows endpoint + key fingerprint so you can verify what is being used.
- Preflights auth with raw HTTP to /documentintelligence/info and /formrecognizer/info
  to distinguish 401 (bad key / local auth disabled) vs 404 (route/version not available).
- Auth modes:
    AZURE_AUTH_MODE=key (default)
    AZURE_AUTH_MODE=aad  (uses azure-identity, requires RBAC on the DI resource)
- Tries FR SDK (azure-ai-formrecognizer) first: prebuilt-document (line.appearance).
- Falls back to DI SDK (azure-ai-documentintelligence) with features=['styleFont'].

Run: streamlit run modules/di_stylefont.py
"""

from __future__ import annotations

import os
import io
import json
import time
import uuid
import base64
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import requests
import streamlit as st


# ---------- env loading ----------
_ENV_SOURCE_HINT = "n/a"
try:
    from dotenv import load_dotenv, find_dotenv
    _dotenv_path = find_dotenv(usecwd=True) or find_dotenv()
    if not _dotenv_path:
        candidate = Path(__file__).resolve().parents[1] / ".env"
        if candidate.exists():
            _dotenv_path = str(candidate)
    if _dotenv_path:
        load_dotenv(_dotenv_path, override=True)  # prefer .env over shell env
        _ENV_SOURCE_HINT = f".env loaded from: {_dotenv_path}"
    else:
        _ENV_SOURCE_HINT = "No .env found via find_dotenv()"
except Exception:
    _ENV_SOURCE_HINT = "python-dotenv not installed or failed to load"

OUT_DIR = Path("outputs/diagnostics/di_stylefont")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _fingerprint(s: str) -> str:
    try:
        return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:8]
    except Exception:
        return "n/a"

def _get_creds() -> Tuple[Optional[str], Optional[str], str, Dict[str, Any]]:
    endpoint = (
        os.getenv("AZURE_DI_ENDPOINT")
        or os.getenv("AZURE_FORMRECOGNIZER_ENDPOINT")
        or os.getenv("FORM_RECOGNIZER_ENDPOINT")
        or os.getenv("COGNITIVE_SERVICE_ENDPOINT")
    )
    key = (
        os.getenv("AZURE_DI_KEY")
        or os.getenv("AZURE_FORMRECOGNIZER_KEY")
        or os.getenv("FORM_RECOGNIZER_KEY")
        or os.getenv("COGNITIVE_SERVICE_KEY")
    )
    mode = (os.getenv("AZURE_AUTH_MODE") or "key").lower().strip()

    # Streamlit secrets fallback
    try:
        if not endpoint and "AZURE_DI_ENDPOINT" in st.secrets:
            endpoint = st.secrets["AZURE_DI_ENDPOINT"]
        if not key and "AZURE_DI_KEY" in st.secrets:
            key = st.secrets["AZURE_DI_KEY"]
    except Exception:
        pass

    diags = {
        "endpoint_present": bool(endpoint),
        "key_present": bool(key),
        "env_source": _ENV_SOURCE_HINT,
        "key_fingerprint": _fingerprint(key or ""),
        "auth_mode": mode,
        "endpoint": endpoint or "?",
    }
    return endpoint, key, mode, diags

def _save_json(payload: Dict[str, Any]) -> str:
    fname = OUT_DIR / f"{int(time.time())}_{uuid.uuid4().hex[:8]}_stylefont.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(fname)

# ---------- preflight auth probe (raw HTTP) ----------
def _http_get(url: str, headers: Dict[str, str]) -> Tuple[int, str]:
    try:
        r = requests.get(url, headers=headers, timeout=20)
        return r.status_code, r.text[:500]
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"

def _preflight_with_key(endpoint: str, key: str) -> Dict[str, Any]:
    """
    Try /documentintelligence/info (v4) and /formrecognizer/info (v3).
    Return codes to help explain 401 vs 404 before SDK calls.
    """
    ep = (endpoint or "").rstrip("/")
    headers = {"Ocp-Apim-Subscription-Key": key}

    di_url = f"{ep}/documentintelligence/info?api-version=2024-07-31"
    fr_url = f"{ep}/formrecognizer/info?api-version=2023-07-31"

    di_code, di_text = _http_get(di_url, headers)
    fr_code, fr_text = _http_get(fr_url, headers)

    return {
        "documentintelligence_info": {"code": di_code, "sample": di_text},
        "formrecognizer_info": {"code": fr_code, "sample": fr_text},
    }

# ---------- SDK paths ----------
def _analyze_with_fr_sdk_key(endpoint: str, key: str, file_bytes: bytes) -> Dict[str, Any]:
    """
    Form Recognizer v3.1 GA path:
    - Calls prebuilt-layout with the STYLE_FONT add-on.
    - Reads result.styles (DocumentStyle[]) just like DI v4.
    """
    try:
        from azure.ai.formrecognizer import DocumentAnalysisClient, DocumentAnalysisFeature
        from azure.core.credentials import AzureKeyCredential
    except Exception as e:
        raise RuntimeError(f"FR SDK not available: {e}")

    client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    # Ask explicitly for the fonts add-on (styleFont) on prebuilt-layout
    poller = client.begin_analyze_document(
        model_id="prebuilt-layout",
        document=io.BytesIO(file_bytes),
        features=[DocumentAnalysisFeature.STYLE_FONT],
    )
    result = poller.result()

    rows: List[Dict[str, Any]] = []
    content = getattr(result, "content", "") or ""
    styles = list(getattr(result, "styles", []) or [])

    for idx, style in enumerate(styles):
        spans = getattr(style, "spans", None) or []
        is_hw = getattr(style, "is_handwritten", None)
        conf = getattr(style, "confidence", None)
        # FR v3.1 exposes the same fields as DI v4 (snake_case in SDK)
        f_style = getattr(style, "font_style", None)     # e.g., "italic"
        f_weight = getattr(style, "font_weight", None)   # e.g., "bold"

        snippet = ""
        if content and spans:
            s0 = spans[0]
            off = getattr(s0, "offset", 0)
            length = getattr(s0, "length", 0)
            try:
                snippet = content[off: off + length]
            except Exception:
                snippet = ""

        rows.append(
            {
                "index": idx,
                "snippet": (snippet or "").replace("\n", " ")[:120],
                "fontStyle": f_style,
                "fontWeight": f_weight,
                "isHandwritten": is_hw,
                "confidence": conf,
                "spanCount": len(spans),
            }
        )

    raw = {
        "kind": "fr_sdk_v3_layout_stylefont",
        "styles_count": len(rows),
        "has_content": bool(content),
    }
    return {"status": "ok" if rows else "no_styles_found", "rows": rows, "raw": raw, "mode": "fr_sdk_v3_layout_stylefont"}

def _analyze_with_di_sdk_key(endpoint: str, key: str, file_bytes: bytes) -> Dict[str, Any]:
    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential
    except Exception as e:
        raise RuntimeError(f"DI SDK not available: {e}")

    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    b64 = base64.b64encode(file_bytes).decode("ascii")
    body = {"base64Source": b64}

    # Some SDK builds accept positional 'body', others need keyword.
    try:
        poller = client.begin_analyze_document("prebuilt-layout", body, features=["styleFont"])
    except TypeError:
        poller = client.begin_analyze_document(model_id="prebuilt-layout", body=body, features=["styleFont"])

    result = poller.result()

    rows: List[Dict[str, Any]] = []
    content = getattr(result, "content", "") or ""
    styles = list(getattr(result, "styles", []) or [])

    for idx, style in enumerate(styles):
        spans = getattr(style, "spans", None) or []
        is_hw = getattr(style, "is_handwritten", None)
        conf = getattr(style, "confidence", None)
        f_style = getattr(style, "font_style", None)
        f_weight = getattr(style, "font_weight", None)

        snippet = ""
        if content and spans:
            s0 = spans[0]
            off = getattr(s0, "offset", s0.get("offset", 0)) if hasattr(s0, "__dict__") or isinstance(s0, dict) else 0
            length = getattr(s0, "length", s0.get("length", 0)) if hasattr(s0, "__dict__") or isinstance(s0, dict) else 0
            try:
                snippet = content[off: off + length]
            except Exception:
                snippet = ""

        rows.append(
            {
                "index": idx,
                "snippet": (snippet or "").replace("\n", " ")[:120],
                "fontStyle": f_style,
                "fontWeight": f_weight,
                "isHandwritten": is_hw,
                "confidence": conf,
                "spanCount": len(spans),
            }
        )

    raw = {
        "kind": "di_sdk_v4",
        "styles_count": len(rows),
        "has_content": bool(content),
    }
    return {"status": "ok" if rows else "no_styles_found", "rows": rows, "raw": raw, "mode": "di_sdk_v4"}

# ---------- Optional AAD (Entra ID) auth ----------
def _get_aad_credential() -> Optional[Any]:
    try:
        from azure.identity import DefaultAzureCredential
        return DefaultAzureCredential(exclude_interactive_browser_credential=False)
    except Exception:
        return None

def _analyze_with_fr_sdk_aad(endpoint: str, cred: Any, file_bytes: bytes) -> Dict[str, Any]:
    from azure.ai.formrecognizer import DocumentAnalysisClient
    client = DocumentAnalysisClient(endpoint=endpoint, credential=cred)
    poller = client.begin_analyze_document("prebuilt-document", io.BytesIO(file_bytes))
    result = poller.result()
    # Reuse shaper
    return _analyze_with_fr_sdk_key(endpoint, "dummy", file_bytes)

def _analyze_with_di_sdk_aad(endpoint: str, cred: Any, file_bytes: bytes) -> Dict[str, Any]:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=cred)
    b64 = base64.b64encode(file_bytes).decode("ascii")
    body = {"base64Source": b64}
    try:
        poller = client.begin_analyze_document("prebuilt-layout", body, features=["styleFont"])
    except TypeError:
        poller = client.begin_analyze_document(model_id="prebuilt-layout", body=body, features=["styleFont"])
    result = poller.result()
    # Reuse shaper
    return _analyze_with_di_sdk_key(endpoint, "dummy", file_bytes)

# ---------- Orchestrator ----------
def analyze_stylefont(file_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    endpoint, key, auth_mode, diags = _get_creds()
    if not endpoint:
        return {"status": "config_error", "rows": [], "raw_result_path": None, "api_version": None,
                "diagnostics": diags, "message": "Missing AZURE_DI_ENDPOINT."}
    endpoint = endpoint.rstrip("/")

    # Preflight (key path only) to explain 401/404 upfront
    preflight = {}
    if auth_mode == "key":
        if not key:
            return {"status": "config_error", "rows": [], "raw_result_path": None, "api_version": None,
                    "diagnostics": diags, "message": "Missing AZURE_DI_KEY for key auth."}
        preflight = _preflight_with_key(endpoint, key)
        diags["preflight"] = preflight

        di_info = preflight["documentintelligence_info"]["code"]
        fr_info = preflight["formrecognizer_info"]["code"]

        # Clear guidance:
        if di_info == 401 and fr_info == 401:
            return {"status": "auth_error", "rows": [], "raw_result_path": None, "api_version": None,
                    "diagnostics": diags,
                    "message": ("401 from both routes. Either the key doesn't belong to this DI resource/region, "
                                "or 'Disable local authentication' is ON. Copy Key 1 from the DI resource's "
                                "Keys & Endpoint blade, or switch AZURE_AUTH_MODE=aad and grant RBAC.")}

    # Auth + run
    try:
        if auth_mode == "aad":
            cred = _get_aad_credential()
            if not cred:
                return {"status": "auth_error", "rows": [], "raw_result_path": None, "api_version": None,
                        "diagnostics": diags,
                        "message": "AZURE_AUTH_MODE=aad but azure-identity is not installed."}

            # Try FR first, then DI
            try:
                out = _analyze_with_fr_sdk_aad(endpoint, cred, file_bytes)
            except Exception:
                out = _analyze_with_di_sdk_aad(endpoint, cred, file_bytes)

        else:
            # key mode: FR first, then DI
            try:
                out = _analyze_with_fr_sdk_key(endpoint, key, file_bytes)
            except Exception as fr_err:
                diags["fr_sdk_error"] = str(fr_err)
                out = _analyze_with_di_sdk_key(endpoint, key, file_bytes)

        raw_path = _save_json(out.get("raw", {}))
        diags.update({"mode": out.get("mode"), "endpoint": endpoint})
        return {"status": out.get("status"), "rows": out.get("rows", []),
                "raw_result_path": raw_path, "api_version": out.get("mode"), "diagnostics": diags}

    except Exception as e:
        diags["final_error"] = str(e)
        return {"status": "sdk_error", "rows": [], "raw_result_path": None, "api_version": None,
                "diagnostics": diags, "message": "Analysis failed. See diagnostics."}

# ---------- UI ----------
def draw_stylefont():
    st.subheader("Azure DI: style detection tester (FR v3 → DI v4)")
    with st.expander("Test (FR appearance or DI styleFont)", expanded=True):
        st.caption("Uses FR v3 line.appearance first; falls back to DI v4 styleFont. Includes auth preflight and diagnostics.")
        colA, colB = st.columns([1, 2])
        with colA:
            uploaded = st.file_uploader("Upload PDF/JPG/PNG", type=["pdf", "jpg", "jpeg", "png"])
        with colB:
            run = st.button("Run style detection", use_container_width=True)

        if run:
            if not uploaded:
                st.warning("Please upload a document first.")
                return
            mime_type = uploaded.type or "application/octet-stream"
            file_bytes = uploaded.read()

            with st.spinner("Analyzing…"):
                result = analyze_stylefont(file_bytes, mime_type)

            diag = result.get("diagnostics", {})
            st.write(
                f"Auth: `{diag.get('auth_mode','key')}` • Endpoint: `{diag.get('endpoint','?')}` • "
                f"Key FP: `{diag.get('key_fingerprint','?')}` • Env: `{diag.get('env_source','n/a')}`"
            )

            pf = diag.get("preflight")
            if pf:
                st.write("Preflight:", pf)

            raw_path = result.get("raw_result_path")
            if raw_path:
                st.success(f"Saved raw snapshot → `{raw_path}`")

            status = result.get("status")
            msg = result.get("message")
            if status == "ok":
                st.success("✅ Styles detected.")
            elif status in ("no_styles_found", "auth_error", "config_error"):
                (st.warning if status != "auth_error" else st.error)(f"{status}: {msg or ''}")
            else:
                st.error(f"Status: {status}. {msg or ''}")
                if "final_error" in diag:
                    st.info(f"Error: {diag['final_error']}")
                if "fr_sdk_error" in diag:
                    st.info(f"FR SDK error: {diag['fr_sdk_error']}")

            rows = result.get("rows", [])
            if rows:
                try:
                    import pandas as pd
                    df = pd.DataFrame(rows, columns=["index","snippet","fontStyle","fontWeight","isHandwritten","confidence","spanCount"])
                    st.dataframe(df, use_container_width=True, height=360)
                except Exception:
                    st.table(rows)
            else:
                st.info("No rows to display.")

# Run directly
if __name__ == "__main__":
    st.set_page_config(page_title="DI styleFont / FR appearance", layout="wide")
    st.title("Azure Document Intelligence: Style Detection")
    draw_stylefont()
