from __future__ import annotations

import io
import json

import pytest
from PIL import Image

from src.review_classifier import (
    Settings,
    _classify_openrouter,
    parse_model_response,
    prepare_image,
    validate_review_text,
)


def make_image(image_format: str = "PNG", size: tuple[int, int] = (1600, 900)) -> bytes:
    image = Image.new("RGB", size, color="teal")
    output = io.BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


def test_validate_review_text_strips_whitespace_and_enforces_word_limit() -> None:
    assert validate_review_text("  item arrived  ", max_words=2) == "item arrived"
    with pytest.raises(ValueError, match="2 words or fewer"):
        validate_review_text("one two three", max_words=2)


def test_validate_review_text_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="Enter review text"):
        validate_review_text("  ", max_words=500)


def test_prepare_image_resizes_and_base64_encodes_png() -> None:
    prepared = prepare_image(make_image(), max_bytes=5 * 1024 * 1024, max_dimension=800)
    with Image.open(io.BytesIO(prepared.data)) as resized:
        assert max(resized.size) == 800
    assert prepared.mime_type == "image/png"
    assert prepared.base64_data


def test_prepare_image_rejects_oversized_and_unsupported_images() -> None:
    image_bytes = make_image()
    with pytest.raises(ValueError, match="MB or smaller"):
        prepare_image(image_bytes, max_bytes=1, max_dimension=800)
    with pytest.raises(ValueError, match="JPG or PNG"):
        prepare_image(make_image("GIF"), max_bytes=5 * 1024 * 1024, max_dimension=800)


def test_parse_model_response_accepts_valid_multimodal_result() -> None:
    response = json.dumps(
        {
            "sentiment": "good",
            "defect_detected": False,
            "visual_text_mismatch": False,
            "reasoning": "The review is positive and the image shows the described item.",
        }
    )
    result = parse_model_response(response, image_attached=True)
    assert result.sentiment == "GOOD"
    assert result.defect_detected is False


def test_parse_model_response_rejects_non_boolean_flags() -> None:
    response = json.dumps(
        {
            "sentiment": "BAD",
            "defect_detected": "false",
            "visual_text_mismatch": False,
            "reasoning": "The review reports a problem.",
        }
    )
    with pytest.raises(ValueError, match="boolean defect and mismatch"):
        parse_model_response(response, image_attached=True)


def test_text_only_response_cannot_claim_visual_findings() -> None:
    response = json.dumps(
        {
            "sentiment": "NEUTRAL",
            "defect_detected": True,
            "visual_text_mismatch": False,
            "reasoning": "A defect is visible.",
        }
    )
    with pytest.raises(ValueError, match="even though no image"):
        parse_model_response(response, image_attached=False)


def test_settings_require_key_for_selected_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    config = Settings.from_env()
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        config.require_api_key()


def test_settings_read_streamlit_secrets_when_env_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    original = __import__("sys").modules.get("streamlit")
    fake_streamlit = type("FakeStreamlit", (), {"secrets": {"OPENAI_API_KEY": "streamlit-key", "LLM_PROVIDER": "openai"}})
    __import__("sys").modules["streamlit"] = fake_streamlit
    try:
        settings = Settings.from_env()
    finally:
        if original is None:
            __import__("sys").modules.pop("streamlit", None)
        else:
            __import__("sys").modules["streamlit"] = original

    assert settings.provider == "openai"
    assert settings.openai_api_key == "streamlit-key"


def test_openrouter_preserves_reasoning_details_for_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    reasoning_details = [{"type": "reasoning.text", "text": "Private model reasoning"}]
    first_message = {
        "content": '{"sentiment":"GOOD","defect_detected":false,'
        '"visual_text_mismatch":false,"reasoning":"The review is positive."}',
        "reasoning_details": reasoning_details,
    }
    final_message = {
        "content": '{"sentiment":"GOOD","defect_detected":false,'
        '"visual_text_mismatch":false,"reasoning":"The review is positive."}'
    }
    request_bodies: list[dict[str, object]] = []

    class FakeResponse:
        def __init__(self, message: dict[str, object]) -> None:
            self.message = message

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"choices": [{"message": self.message}]}

    def fake_post(url: str, *, headers: dict[str, str], json: dict[str, object], timeout: float) -> FakeResponse:
        assert url == "https://openrouter.ai/api/v1/chat/completions"
        assert headers["Authorization"] == "Bearer test-key"
        request_bodies.append(json)
        return FakeResponse(first_message if len(request_bodies) == 1 else final_message)

    monkeypatch.setattr("requests.post", fake_post)
    settings = Settings.from_env()
    settings = Settings(
        **{
            **settings.__dict__,
            "provider": "openrouter",
            "openrouter_api_key": "test-key",
            "openrouter_verify_response": True,
        }
    )

    result = _classify_openrouter("Great product", None, settings, "test-key")

    assert result == final_message["content"]
    assert len(request_bodies) == 2
    second_messages = request_bodies[1]["messages"]
    assert second_messages[-2]["reasoning_details"] is reasoning_details
    assert request_bodies[0]["reasoning"] == {"enabled": settings.openrouter_reasoning_enabled}