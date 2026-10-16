# Review Lens

A Streamlit demo for multimodal e-commerce review classification. It combines review text with an optional product image, then returns sentiment (`GOOD`, `NEUTRAL`, or `BAD`), a visible-defect flag, a text-image mismatch flag, and a short evidence-based explanation. If no image is uploaded, the app performs text-only classification.

## Live links

- Interactive classifier: [Open Review Lens](https://review-classifier-inwgozhfxisaxxummuwaz3.streamlit.app/)
- Static project requirements: [Open the PRD on GitHub Pages](https://mandalbikash324-cell.github.io/review-classifier/)
- Complete text-based source code and configuration: [Download the project source PDF](./Review_Lens_Full_Project_Source.pdf)

The Streamlit app is connected to this repository's `main` branch and `app.py`; pushes to the branch are deployed by Streamlit Community Cloud. GitHub Pages automatically publishes the static PRD when changes are pushed to `main`.

## Requirements

- Python 3.10 or newer
- An API key for OpenAI, Gemini, or OpenRouter
- Internet access for API requests

The app supports OpenAI, Gemini, and OpenRouter. Model access, image support, rate limits, and prices depend on your provider account and model availability.

## Setup

From the project directory:

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Edit the included local `.env` (or copy `.env.example` to `.env`) and add the API key for the selected provider. OpenRouter is selected by default:

```dotenv
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your-openrouter-api-key
OPENROUTER_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
```

The Python request code is in `src/review_classifier.py`; `.env` contains only settings. Replace the empty `OPENROUTER_API_KEY=` value with your actual key locally. Do not paste the literal `<OPENROUTER_API_KEY>` placeholder, and do not share or commit the real key.

The model in the example is the one from the supplied snippet. Check OpenRouter for current availability and vision support; choose a vision-capable model when uploading images. For direct OpenAI:

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=your-key-here
```

For Gemini, set `LLM_PROVIDER=gemini` and provide `GEMINI_API_KEY`. Provider selection, model names, reasoning behavior, limits, and request timeout are configured in `.env`. Do not commit `.env` or share API keys.

In VS Code, open the command palette, choose **Python: Select Interpreter**, and select `.venv/bin/python` (Windows: `.venv\\Scripts\\python.exe`).

## Run

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`. Enter a review, optionally add a JPG/PNG image (up to the configured limit), then select **Classify review**.

## GitHub Pages

The GitHub Pages workflow publishes `PRD.html` as the static site homepage when changes are pushed to `main`, or when the workflow is run manually. The page is available at `https://mandalbikash324-cell.github.io/review-classifier/` after the first successful deployment.

GitHub Pages cannot run the interactive Streamlit classifier or its Python API requests. Use a Python hosting platform such as Streamlit Community Cloud to publish the full application.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` | `openai` or `gemini` |
| `OPENAI_API_KEY` | empty | OpenAI API credential |
| `GEMINI_API_KEY` | empty | Gemini API credential |
| `OPENROUTER_API_KEY` | empty | OpenRouter API credential |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI vision-capable model available to your account |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini vision-capable model available to your account |
| `OPENROUTER_MODEL` | `nvidia/nemotron-3-ultra-550b-a55b:free` | OpenRouter model id; verify vision support for image requests |
| `OPENROUTER_REASONING_ENABLED` | `true` | Include OpenRouter's reasoning-enabled option |
| `OPENROUTER_VERIFY_RESPONSE` | `true` | Make a second request to verify the result, preserving `reasoning_details` unchanged; increases latency and request usage |
| `OPENROUTER_HTTP_REFERER` | empty | Optional OpenRouter attribution header |
| `OPENROUTER_APP_TITLE` | `Review Lens` | OpenRouter app-title header |
| `API_TIMEOUT_SECONDS` | `20` | Request timeout; this is separate from the 3.5-second PRD latency target |
| `MAX_REVIEW_WORDS` | `500` | Maximum review length in words |
| `MAX_IMAGE_MB` | `5` | Maximum uploaded image size |
| `MAX_IMAGE_DIMENSION` | `1024` | Longest resized image dimension in pixels |

Images are validated as JPG/PNG, resized while preserving aspect ratio, and encoded for API transport. OpenAI receives a Base64 data URL. The Gemini SDK serializes the image bytes as inline image data. Images are not saved by the app.

## Tests

```bash
pytest
```

The unit tests cover input limits, image validation/resizing, response parsing, text-only safeguards, and OpenRouter reasoning-detail handoff without making API calls.

## Evaluation notes

The included dependencies support later evaluation with labeled data and scikit-learn. Report macro F1, per-class precision/recall, confusion matrices, API latency, model/provider, and observed cost. The PRD's F1, latency, and cost figures are targets to measure on a documented held-out dataset, not guaranteed model performance. API-based image-only ablation requires an image-labeled evaluation set; text-only ablation should omit images.

## Project structure

```text
app.py                      Streamlit user interface
src/review_classifier.py    Input handling, image preprocessing, provider calls, response validation
tests/test_review_classifier.py  Focused backend tests
.env                        Local ignored configuration
.env.example                Safe configuration template
requirements.txt            Python dependencies
```