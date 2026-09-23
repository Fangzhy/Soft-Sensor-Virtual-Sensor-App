"""Read server-side secrets, independently of the working directory."""

import os
from pathlib import Path

from dotenv import dotenv_values

# Verified in OpenRouter's public catalog: zero prompt/completion pricing,
# structured outputs, and optional reasoning. Avoid random classifier routing.
DEFAULT_FREE_MODEL = "nex-agi/nex-n2.5-mini:free"


def openrouter_settings() -> tuple[str, str]:
    """Process environment overrides the explicit root .env, never docs/.env.

    Read at request time so a corrected key can be used without printing it or
    exposing it to the browser. Cloud root-level secrets use the environment. Disable interpolation of unrelated env values.
    """
    values = dotenv_values(Path(__file__).resolve().parents[1] / ".env", interpolate=False)
    # Preserve the user's existing OPEN_ROUTER_API setting as a supported alias.
    key = os.getenv("OPENROUTER_API_KEY", os.getenv(
        "OPEN_ROUTER_API", values.get("OPENROUTER_API_KEY") or values.get("OPEN_ROUTER_API") or "",
    )).strip()
    model = os.getenv("OPENROUTER_MODEL", values.get("OPENROUTER_MODEL") or DEFAULT_FREE_MODEL).strip()
    return key, model
