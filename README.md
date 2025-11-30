# Liminal Backrooms (Free AI Fork)

> A fork of the original **Liminal Backrooms** refit to run entirely on free AI APIs (OpenRouter free models + Cloudflare Workers AI). Multi-agent conversations, images, and optional video inside a dark, liminal GUI.

## Highlights
- **Free-first stack**: Defaults to free OpenRouter model variants and Cloudflare Workers AI so you can explore without paid keys.
- **Multi-agent chaos**: Up to 5 AI participants that can invite each other, mute, and swap personas mid-run.
- **Visuals on tap**: `!image` pipes through Workers AI; HTML exports include generated art.
- **Optional Sora video**: `!video` support is wired but disabled by default; add an OpenAI key if you want it.
- **Scenario library**: Backrooms exploration plus group chat, Slack chaos, and other prebuilt vibes.

## Quickstart
```bash
git clone <repo-url>
cd liminal_backrooms
poetry install
cp .env.example .env  # or create your own
poetry run python main.py
```

## Environment
- Core keys (free-friendly):
  - `OPENROUTER_API_KEY` — routes all LLM calls; defaults use free OpenRouter models.
  - `CF_ACCOUNT_ID`, `CF_AI_TOKEN` — required for `!image` via Cloudflare Workers AI (cheap/free tiers available).
  - `CF_AI_MODEL` — defaults to `@cf/stabilityai/stable-diffusion-xl-base-1.0` (budget-friendly).
  - `CF_AI_DAILY_LIMIT` — local guardrail to cap image generations per UTC day.
- Optional extras:
  - `OPENAI_API_KEY` — only for Sora `!video`.
  - `GEMINI_API_KEY`, `GROQ_API_KEY`, `CEREBRAS_API_KEY` — use direct vendor routes if you prefer.

Use `.env.example` as a template. Python 3.10–3.11 and Poetry are required; tested on Windows 10/11 and modern Linux.

## How to Use
1. Launch: `poetry run python main.py`
2. In the GUI:
   - Pick **AI-AI** or **Human-AI** mode and set turn count (1–100).
   - Choose models (config defaults point to free OpenRouter IDs).
   - Select a scenario prompt and hit start.
   - Use `!image "desc"` to verify Cloudflare keys (e.g., `!image "a liminal hallway bathed in neon"`).
   - Export conversations (HTML + images) from the interface.

## Command Cheat Sheet
- `!add_ai "Model Name" "persona"` — invite another AI (max 5 total).
- `!image "description"` — generate an image via Workers AI.
- `!video "description"` — Sora video (disabled in shipped scenarios; needs OpenAI key).
- `!mute_self` — sit out a turn.

## Config Touchpoints
- `config.py`: tweak turn delays, toggle chain-of-thought display, and manage available models.
- `SYSTEM_PROMPT_PAIRS` in `config.py`: add or edit scenario prompts for AI-1 through AI-5.
- `AI_MODELS` in `config.py`: map display names to OpenRouter (or vendor) model IDs; defaults lean on free endpoints.

## Troubleshooting
- Empty or slow replies: confirm `OPENROUTER_API_KEY` is set and the selected model is available.
- Images failing: check `CF_ACCOUNT_ID`, `CF_AI_TOKEN`, and your daily limit.
- GUI issues: ensure PyQt6 installed via `poetry install`; verify Python version is 3.10–3.11.

## Credits & License
- Forked from the original **Liminal Backrooms** project, reworked for all-free API usage.
- MIT License — see `LICENSE` for full text.
