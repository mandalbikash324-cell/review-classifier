from __future__ import annotations

import time

import streamlit as st
from dotenv import load_dotenv

from src.review_classifier import Settings, classify_review

load_dotenv()
settings = Settings.from_env()

st.set_page_config(
    page_title="Review Lens | Multimodal Classifier",
    page_icon="R",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root { --teal: #087e78; --ink: #1d2935; --muted: #526373; }
    .block-container { max-width: 1120px; padding-top: 2.2rem; }
    .eyebrow { color: var(--teal); font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
    .subtle { color: var(--muted); }
    div[data-testid="stMetric"] { background: #f2f7f6; border-left: 3px solid var(--teal); padding: .75rem 1rem; }
    div.stButton > button[kind="primary"] { background: var(--teal); border-color: var(--teal); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">Multimodal review intelligence</div>', unsafe_allow_html=True)
st.title("Review Lens")
st.markdown(
    '<p class="subtle">Classify review sentiment and inspect product evidence in one pass.</p>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.15, 0.85], gap="large")
with left:
    review_text = st.text_area(
        "Customer review",
        placeholder="Paste the customer's review here...",
        height=220,
        max_chars=6000,
        help=f"Up to {settings.max_review_words} words.",
    )
    words = len(review_text.split())
    st.caption(f"{words} / {settings.max_review_words} words")

with right:
    uploaded_image = st.file_uploader(
        "Product image (optional)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        help=f"JPG or PNG, up to {settings.max_image_bytes // (1024 * 1024)} MB.",
    )
    if uploaded_image is not None:
        st.image(uploaded_image, caption=uploaded_image.name, use_container_width=True)

provider_label = {
    "openai": "OpenAI",
    "gemini": "Gemini",
    "openrouter": "OpenRouter",
}[settings.provider]
st.caption(f"Provider: {provider_label} · Model: {settings.model}")
classify_clicked = st.button("Classify review", type="primary", icon=":material/search:")

if classify_clicked:
    image_data = uploaded_image.getvalue() if uploaded_image is not None else None
    started_at = time.perf_counter()
    try:
        with st.spinner("Analyzing review and image evidence..."):
            result = classify_review(review_text, image_data, settings=settings)
        elapsed = time.perf_counter() - started_at
        st.divider()
        st.subheader("Classification")
        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("Sentiment", result.sentiment)
        metric2.metric("Visible defect", "Detected" if result.defect_detected else "Not detected")
        metric3.metric("Text-image mismatch", "Flagged" if result.visual_text_mismatch else "Not flagged")
        metric4.metric("Response time", f"{elapsed:.2f} s")
        st.markdown("**Reasoning**")
        st.write(result.reasoning)
        if elapsed > 3.5:
            st.caption("The PRD latency target is 3.5 seconds; actual API time depends on provider and network conditions.")
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error("Classification request failed. Check the provider, API key, model name, and network connection.")
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")

st.divider()
st.caption("Visual flags are based on the uploaded image only. A missing image triggers text-only classification.")