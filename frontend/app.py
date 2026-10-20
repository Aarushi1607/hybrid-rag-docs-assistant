from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
ASK_TIMEOUT_S = float(os.getenv("ASK_TIMEOUT_S", "180"))  # local LLMs are slow
HEALTH_TIMEOUT_S = 3.0

EXAMPLES = [
    "How do I define a path operation using @app.get()?",
    "How do I create a POST endpoint using @app.post()?",
    "What is a path operation function in FastAPI?"
]

PLACEHOLDER = "Ask anything about FastAPI, e.g. How do I create a POST endpoint?"

MODES = {
    "hybrid": "Hybrid: Dense + BM25 + RRF",
    "dense": "Dense only",
}
MODE_HELP = {
    "hybrid": "Combines semantic (ChromaDB) and keyword (BM25) results with "
    "weighted Reciprocal Rank Fusion (0.7 dense / 0.3 sparse).",
    "dense": "Uses only semantic vector search over ChromaDB. Faster to reason "
    "about, but can miss exact keyword matches.",
}

# Semantic colors used from Python (layout colors live in the CSS variables below).
COLORS = {"online": "#5E8C61", "offline": "#B5524A"}
CONFIDENCE = {  # (text, tint) - understated, readable on white
    "high": ("#4F7A52", "#EAF1E8"),
    "medium": ("#8F6A1C", "#F6EFDD"),
    "low": ("#A8473F", "#F6E6E3"),
    "none": ("#77796F", "#EFEDE6"),
}
ICON_BOOK = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#66775C" '
    'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v16H6.5A2.5 2.5 0 0 0 4 21.5z"/>'
    '<path d="M4 21.5V5.5"/><path d="M9 8h7"/></svg>'
)
ICON_DOC = ICON_BOOK.replace('width="22" height="22"', 'width="28" height="28"')

st.set_page_config(
    page_title="DocuMind",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed",
)


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap');
:root {
  --bg:#F7F6F2; --text:#252720; --muted:#77796F; --card:#FFFFFF; --border:#E7E5DE;
  --sage:#75866A; --sage-deep:#66775C; --sage-tint:#E9EDE4; --terra:#C58B70; --radius:12px;
}
html, body, .stApp { background:var(--bg); color:var(--text); }
.stApp, .stApp button, .stApp input, .stApp textarea, .stApp summary {
  font-family:'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
}
header[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer, [data-testid="stSidebar"], [data-testid="collapsedControl"] { display:none; }
.block-container { max-width:760px; padding-top:1.6rem; padding-bottom:4rem; }

/* Text */
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color:var(--text); line-height:1.65; }
[data-testid="stCaptionContainer"], .dm-muted, .dm-meta { color:var(--muted); }
[data-testid="stMarkdownContainer"] code {
  background:#F0EEE7; color:var(--text); border-radius:6px; padding:.1rem .35rem; font-size:.88em;
}
[data-testid="stMarkdownContainer"] pre code { background:transparent; padding:0; }
[data-testid="stMarkdownContainer"] pre, [data-testid="stCode"] pre {
  background:#F3F1EA !important; border:1px solid var(--border); border-radius:10px;
}

/* Header */
.dm-brand { display:flex; align-items:center; gap:.5rem; }
.dm-word { font-family:'Newsreader', Georgia, serif; font-size:1.45rem; font-weight:500; letter-spacing:-.01em; }
.dm-desc { color:var(--muted); font-size:.85rem; margin-top:.1rem; }
.dm-status { display:flex; justify-content:flex-end; align-items:center; gap:.7rem; flex-wrap:wrap; }
.dm-pill {
  display:inline-flex; align-items:center; gap:.45rem; background:var(--card);
  border:1px solid var(--border); border-radius:999px; padding:.22rem .7rem; font-size:.8rem;
}
.dm-dot { width:.5rem; height:.5rem; border-radius:50%; display:inline-block; }
.dm-meta { font-size:.8rem; }
.dm-rule { border-top:1px solid var(--border); margin:1rem 0 2.2rem; }
@media (max-width:640px){ .dm-status { justify-content:flex-start; margin-top:.4rem; } }

/* Intro */
.dm-h1 { font-family:'Newsreader', Georgia, serif; font-size:2rem; font-weight:400; letter-spacing:-.015em; margin:0 0 .35rem; line-height:1.2; }
.dm-lede { color:var(--muted); margin:0 0 1.2rem; font-size:.98rem; }

/* Question form = the visual focus */
[data-testid="stForm"] {
  background:var(--card); border:1px solid var(--border); border-radius:14px; padding:.9rem;
}
.stTextInput input {
  background:var(--card); color:var(--text); border:1px solid var(--border);
  border-radius:10px; padding:.8rem .95rem; font-size:1rem;
}
.stTextInput input::placeholder { color:var(--muted); opacity:1; }
.stTextInput input:focus { border-color:var(--sage); box-shadow:0 0 0 1px var(--sage); }

/* Buttons */
.stFormSubmitButton > button, .stButton > button {
  border-radius:10px; border:1px solid var(--border); background:var(--card);
  color:var(--text); font-weight:500; transition:border-color .15s, background .15s;
}
.stFormSubmitButton > button[kind="primaryFormSubmit"] {
  background:var(--sage-deep); border-color:var(--sage-deep); color:#fff; font-weight:600; padding:.55rem 1rem;
}
.stFormSubmitButton > button[kind="primaryFormSubmit"]:hover { background:#58684F; border-color:#58684F; color:#fff; }
.stFormSubmitButton > button:focus-visible, .stButton > button:focus-visible { outline:2px solid var(--sage); outline-offset:2px; }

/* Example chips */
.st-key-examples .stButton > button {
  background:transparent; border-radius:999px; font-size:.82rem; color:var(--muted);
  padding:.25rem .8rem; min-height:0;
}
.st-key-examples .stButton > button:hover { border-color:var(--sage); color:var(--text); background:var(--sage-tint); }

/* Segmented retrieval control */
.st-key-modebox div[role="radiogroup"] {
  display:inline-flex; gap:.2rem; background:#EFEDE6; border-radius:10px; padding:.2rem;
}
.st-key-modebox div[role="radiogroup"] > label { padding:.3rem .8rem; border-radius:8px; margin:0; font-size:.88rem; }
.st-key-modebox div[role="radiogroup"] > label > div:first-child { display:none; }
.st-key-modebox div[role="radiogroup"] > label:has(input:checked) {
  background:var(--card); box-shadow:0 0 0 1px var(--sage);
}
.st-key-modebox div[role="radiogroup"] > label:has(input:checked) p { color:var(--text); font-weight:600; }
.st-key-modebox div[role="radiogroup"] > label p { color:var(--muted); }

/* Answer */
.dm-sec { display:flex; justify-content:space-between; align-items:center; margin:2rem 0 .6rem; }
.dm-sec-title { font-family:'Newsreader', Georgia, serif; font-size:1.35rem; font-weight:500; }
.dm-badge {
  display:inline-flex; align-items:center; gap:.4rem; border-radius:999px;
  padding:.15rem .65rem; font-size:.78rem; font-weight:500;
}
.st-key-answer_card {
  background:var(--card); border:1px solid var(--border); border-radius:14px; padding:1.1rem 1.4rem;
}
.st-key-answer_card [data-testid="stMarkdownContainer"] p { font-size:1rem; }

/* Expanders (sources, details, copy) */
[data-testid="stExpander"] { background:var(--card); border:1px solid var(--border); border-radius:10px; margin-bottom:.4rem; }
[data-testid="stExpander"] summary { font-size:.9rem; color:var(--text); }
[data-testid="stExpander"] summary:hover { color:var(--sage-deep); }
.dm-preview { color:#4A4C43; white-space:pre-wrap; font-size:.9rem; line-height:1.6; }

/* Empty state */
.dm-empty { text-align:center; padding:2.8rem 1rem; color:var(--muted); }
.dm-empty b { display:block; color:var(--text); font-family:'Newsreader', Georgia, serif; font-size:1.2rem; font-weight:500; margin:.6rem 0 .25rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)



class ApiError(Exception):
    """A failure with a short, developer-friendly message."""


@dataclass
class Source:
    index: int
    name: str | None = None
    score: float | None = None
    preview: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Answer:
    text: str
    confidence_label: str | None
    confidence_score: float | None
    sources: list[Source]
    mode: str
    elapsed_s: float
    raw: dict[str, Any]


def _first(d: dict, keys: tuple[str, ...]) -> Any:
    """Return the first present, non-None value among candidate keys."""
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    return None


def _as_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def parse_source(i: int, item: Any) -> Source:
    if isinstance(item, str):  # backend returned plain text chunks
        return Source(index=i, preview=item)
    if not isinstance(item, dict):
        return Source(index=i)
    meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    name = _first(item, ("filename", "file", "source", "path", "doc", "title")) or _first(
        meta, ("filename", "file", "source", "path", "title")
    )
    return Source(
        index=_first(item, ("index", "id", "rank")) or i,
        name=str(name) if name else None,
        score=_as_float(_first(item, ("score", "relevance_score", "rrf_score", "relevance"))),
        preview=_first(item, ("preview", "text", "content", "chunk", "snippet")),
    )


def parse_answer(data: Any, mode: str, elapsed: float) -> Answer:
    if not isinstance(data, dict):
        raise ApiError("Unexpected response: expected a JSON object.")
    text = _first(data, ("answer", "response", "generated_answer"))
    if text is None:
        raise ApiError(
            f"Response is missing an answer field. Keys received: {sorted(data.keys())}"
        )

    conf = data.get("confidence")
    label = score = None
    if isinstance(conf, dict):
        label = _first(conf, ("label", "level"))
        score = _as_float(_first(conf, ("top_score", "score", "value")))
    else:  # flat shapes: "confidence": 0.82 / "confidence": "high"
        label = _first(data, ("confidence_label", "confidence_level"))
        score = _as_float(_first(data, ("confidence_score",)))
        if score is None and label is None and conf is not None:
            if _as_float(conf) is not None:
                score = _as_float(conf)
            else:
                label = str(conf)

    raw_sources = _first(data, ("sources", "retrieved_sources", "citations")) or []
    if not isinstance(raw_sources, list):
        raise ApiError("'sources' is present but is not a list.")
    sources = [parse_source(i, s) for i, s in enumerate(raw_sources, start=1)]

    return Answer(
        text=str(text),
        confidence_label=str(label).lower() if label else None,
        confidence_score=score,
        sources=sources,
        mode=str(data.get("retrieval_mode") or mode),
        elapsed_s=elapsed,
        raw=data,
    )


@st.cache_data(ttl=5, show_spinner=False)
def fetch_health(base_url: str) -> dict[str, Any]:
    """Real call to GET /v1/health. Cached for 5s to avoid hammering on reruns."""
    try:
        r = requests.get(f"{base_url}/v1/health", timeout=HEALTH_TIMEOUT_S)
        r.raise_for_status()
        body = r.json() if r.content else {}
        return {"ok": True, "body": body if isinstance(body, dict) else {}}
    except requests.ConnectionError:
        return {"ok": False, "error": f"Cannot reach {base_url}. Is FastAPI running?"}
    except requests.Timeout:
        return {"ok": False, "error": "Health check timed out."}
    except requests.HTTPError as e:
        return {"ok": False, "error": f"Health check returned HTTP {e.response.status_code}."}
    except ValueError:
        return {"ok": False, "error": "Health check did not return valid JSON."}


def ask_backend(question: str, mode: str) -> Answer:
    payload = {"question": question, "retrieval_mode": mode}
    start = time.perf_counter()
    try:
        r = requests.post(f"{API_BASE_URL}/v1/ask", json=payload, timeout=ASK_TIMEOUT_S)
    except requests.ConnectionError:
        raise ApiError(f"Backend unreachable at {API_BASE_URL}. Start FastAPI and retry.")
    except requests.Timeout:
        raise ApiError(
            f"Request timed out after {ASK_TIMEOUT_S:.0f}s. Ollama may still be "
            "loading the model; try again or raise ASK_TIMEOUT_S."
        )
    elapsed = time.perf_counter() - start

    if not r.ok:
        detail = ""
        try:
            body = r.json()
            detail = body.get("detail", "") if isinstance(body, dict) else ""
        except ValueError:
            pass
        raise ApiError(f"API returned HTTP {r.status_code}. {str(detail)[:300]}".strip())
    try:
        data = r.json()
    except ValueError:
        raise ApiError("API returned a non-JSON response.")
    return parse_answer(data, mode, elapsed)

def init_state() -> None:
    st.session_state.setdefault("question", "")
    st.session_state.setdefault("mode", "hybrid")
    st.session_state.setdefault("result", None)  # Answer | None
    st.session_state.setdefault("error", None)  # str | None
    st.session_state.setdefault("asked_question", None)


def set_example(text: str) -> None:
    st.session_state.question = text  # runs before widgets re-instantiate

def confidence_color(label: str | None) -> tuple[str, str] | None:
    """Return (text, tint) for a confidence label, or None when unknown."""
    if not label:
        return None
    for key in ("high", "medium", "low", "none"):
        if label.startswith(key[:3]):
            return CONFIDENCE[key]
    return None


def render_header() -> None:
    health = fetch_health(API_BASE_URL)
    left, right = st.columns([3, 2], vertical_alignment="center")
    with left:
        st.markdown(
            f'<div class="dm-brand">{ICON_BOOK}<span class="dm-word">DocuMind</span></div>'
            '<div class="dm-desc">Your AI-powered FastAPI documentation companion.</div>',
            unsafe_allow_html=True,
        )
    with right:
        ok = health["ok"]
        color = COLORS["online"] if ok else COLORS["offline"]
        html = (
            f'<span class="dm-pill"><span class="dm-dot" style="background:{color}"></span>'
            f'{"Backend online" if ok else "Backend offline"}</span>'
        )
        if ok:
            chunks = _first(health["body"], ("chunks_loaded", "chunks", "num_chunks"))
            if chunks is not None:
                html += f'<span class="dm-meta">{_escape(str(chunks))} chunks indexed</span>'
        st.markdown(f'<div class="dm-status">{html}</div>', unsafe_allow_html=True)
        if not ok:
            st.markdown(
                f'<div class="dm-status dm-meta" style="margin-top:.3rem">{_escape(health["error"])}</div>',
                unsafe_allow_html=True,
            )
    st.markdown('<div class="dm-rule"></div>', unsafe_allow_html=True)


def render_input() -> tuple[bool, str]:
    st.markdown(
        '<p class="dm-h1">What are you trying to build?</p>'
        '<p class="dm-lede">Answers are grounded in the FastAPI documentation, with the sources shown alongside.</p>',
        unsafe_allow_html=True,
    )
    with st.form("ask_form"):
        c_in, c_btn = st.columns([5, 1.7], vertical_alignment="center")
        with c_in:
            st.text_input("Question", key="question", placeholder=PLACEHOLDER, label_visibility="collapsed")
        with c_btn:
            submitted = st.form_submit_button("Ask question", type="primary", use_container_width=True)

    with st.container(key="examples"):
        st.markdown('<p class="dm-meta" style="margin:.8rem 0 .3rem">Try an example</p>', unsafe_allow_html=True)
        cols = st.columns(len(EXAMPLES))
        for col, ex in zip(cols, EXAMPLES):
            col.button(ex, key=f"ex_{ex}", on_click=set_example, args=(ex,), use_container_width=True)
    return submitted, st.session_state.question


def render_mode_selector() -> None:
    st.markdown('<p class="dm-meta" style="margin:1.1rem 0 .35rem">Retrieval mode</p>', unsafe_allow_html=True)
    with st.container(key="modebox"):
        st.radio(
            "Retrieval mode",
            options=list(MODES),
            format_func=MODES.get,
            key="mode",
            horizontal=True,
            label_visibility="collapsed",
        )
    st.caption(MODE_HELP[st.session_state.mode])


def render_answer(res: Answer) -> None:
    badge = ""
    colors = confidence_color(res.confidence_label)
    if res.confidence_label or res.confidence_score is not None:
        fg, bg = colors or CONFIDENCE["none"]
        text = f"{res.confidence_label.capitalize()} confidence" if res.confidence_label else "Confidence"
        if res.confidence_score is not None:
            text += f" {res.confidence_score:.2f}"
        badge = (
            f'<span class="dm-badge" style="color:{fg};background:{bg}">'
            f'<span class="dm-dot" style="background:{fg};width:.4rem;height:.4rem"></span>{_escape(text)}</span>'
        )
    st.markdown(
        f'<div class="dm-sec"><span class="dm-sec-title">Answer</span>{badge}</div>',
        unsafe_allow_html=True,
    )

    if not res.text.strip():
        st.warning("The backend returned an empty answer. Check the generation step and Ollama logs.")
        return

    with st.container(key="answer_card"):
        st.markdown(res.text)  # Markdown rendered natively

    if res.confidence_label and res.confidence_label.startswith("low"):
        st.warning(
            "Retrieval confidence is low. Treat this response with caution and "
            "check the FastAPI documentation directly."
        )
    with st.expander("Copy answer"):
        st.code(res.text, language=None, wrap_lines=True)  # built-in copy button


def render_sources(res: Answer) -> None:
    if not res.sources:
        if res.confidence_label in ("low", "none"):
            st.info("No sources shown: retrieval confidence was too low to answer from the documentation.")
        else:
            st.info("The backend returned no sources for this answer.")
        return
    st.markdown(
        f'<div class="dm-sec"><span class="dm-sec-title">Sources used</span>'
        f'<span class="dm-meta">{len(res.sources)} retrieved</span></div>',
        unsafe_allow_html=True,
    )
    for s in res.sources:  # preserve backend order
        title = f"[{s.index}]  {s.name or 'Unnamed source'}"
        if s.score is not None:
            title += f"   (score {s.score:.4f})"
        with st.expander(title):
            if s.preview:
                text = str(s.preview)
                st.markdown(
                    f'<div class="dm-preview">{_escape(text[:700])}{"…" if len(text) > 700 else ""}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No preview text provided for this source.")


def render_details(res: Answer) -> None:
    st.write("")
    with st.expander("Retrieval details"):
        st.markdown(f"- **Mode:** {MODES.get(res.mode, res.mode)}")
        st.markdown(f"- **Sources returned:** {len(res.sources)}")
        scores = [s.score for s in res.sources if s.score is not None]
        if scores:
            st.markdown(f"- **Top relevance score:** {max(scores):.4f}")
        st.markdown(f"- **Response time:** {res.elapsed_s:.1f}s")
        st.caption(
            "Confidence comes from the top dense (vector) similarity score. "
            "Source scores come from the selected retrieval mode, so they are not "
            "directly comparable between Hybrid and Dense."
        )
    # Sibling, not nested: Streamlit does not allow an expander inside an expander.
    with st.expander("Raw response"):
        st.json(res.raw)


def _escape(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_empty_state() -> None:
    st.markdown(
        f'<div class="dm-empty">{ICON_DOC}<b>Ask your first question</b>'
        "Pick an example above or type your own to get an answer with sources.</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    init_state()
    render_header()
    st.write("")
    submitted, question = render_input()
    render_mode_selector()

    if submitted:
        q = question.strip()
        if not q:
            st.session_state.error = "Enter a question before submitting."
            st.session_state.result = None
        else:
            with st.spinner("Retrieving context and generating answer…"):
                try:
                    st.session_state.result = ask_backend(q, st.session_state.mode)
                    st.session_state.error = None
                    st.session_state.asked_question = q
                except ApiError as e:
                    st.session_state.result = None
                    st.session_state.error = str(e)

    st.write("")
    if st.session_state.error:
        st.error(st.session_state.error)
    elif st.session_state.result is None:
        render_empty_state()
    else:
        res: Answer = st.session_state.result
        render_answer(res)
        render_sources(res)
        render_details(res)


main()