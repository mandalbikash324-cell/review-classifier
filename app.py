from __future__ import annotations

import time

import streamlit as st
from dotenv import load_dotenv

from src.review_classifier import Settings, classify_review

load_dotenv()
settings = Settings.from_env()

st.set_page_config(
    page_title="Review Lens | Review intelligence",
    page_icon=":material/auto_awesome:",
    layout="wide",
)

st.badge(
    "MULTIMODAL REVIEW INTELLIGENCE",
    icon=":material/auto_awesome:",
    color="violet",
)
st.title("Review Lens")
st.markdown(
    "Understand the story behind every review with AI-powered sentiment and visual analysis."
)

provider_label = {
    "openai": "OpenAI",
    "gemini": "Gemini",
    "openrouter": "OpenRouter",
}[settings.provider]

with st.container(border=True):
    st.subheader("Analyze a review", icon=":material/rate_review:")
    st.caption(
        "Add the customer's words and, if available, a product photo. "
        "The image is optional."
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
            st.image(uploaded_image, caption=uploaded_image.name, width="stretch")

    classify_clicked = st.button(
        "Detect and classify",
        type="primary",
        icon=":material/search:",
        width="stretch",
    )

st.caption(f"Powered by {provider_label} · {settings.model}")

if classify_clicked:
    image_data = uploaded_image.getvalue() if uploaded_image is not None else None
    started_at = time.perf_counter()
    try:
        with st.spinner("Analyzing review and image evidence..."):
            result = classify_review(review_text, image_data, settings=settings)
        elapsed = time.perf_counter() - started_at
        sentiment_colors = {"GOOD": "green", "NEUTRAL": "orange", "BAD": "red"}
        with st.container(border=True):
            st.subheader("Analysis results", icon=":material/insights:")
            if uploaded_image is not None:
                st.badge(
                    f"Detected product: {result.product_type}",
                    icon=":material/category:",
                    color="blue",
                )
                st.space("small")
            st.badge(
                f"{result.sentiment} sentiment",
                icon=":material/mood:",
                color=sentiment_colors[result.sentiment],
            )
            st.space("small")
            metric1, metric2, metric3 = st.columns(3)
            metric1.metric(
                "Visible defect",
                "Detected" if result.defect_detected else "Not detected",
            )
            metric2.metric(
                "Review–image match",
                "Mismatch" if result.visual_text_mismatch else "Aligned",
            )
            metric3.metric("Response time", f"{elapsed:.2f} s")

            st.subheader("Evidence summary", icon=":material/format_quote:")
            st.write(result.reasoning)
        if elapsed > 3.5:
            st.caption(
                "The PRD latency target is 3.5 seconds; actual API time depends "
                "on provider and network conditions."
            )
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(
            "Analysis failed. Check the provider, API key, model name, "
            "and network connection."
        )
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")

st.caption(
    "Visual flags are based on the uploaded image only. "
    "Without an image, Review Lens performs text-only classification."
)