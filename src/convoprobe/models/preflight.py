"""Check a local Ollama server is up and has the needed models before a run."""

from __future__ import annotations

import httpx

from convoprobe.errors import ConvoprobeError


async def check_ollama(
    base_url: str,
    required_models: list[str],
    transport: httpx.AsyncBaseTransport | None = None,
) -> None:
    """Raise ConvoprobeError with a fix-it message if Ollama or a model is missing."""
    unreachable = ConvoprobeError(
        f"Ollama is not reachable at {base_url}. Start Ollama, or rerun with --replay <cassette>."
    )
    try:
        async with httpx.AsyncClient(transport=transport, timeout=3.0) as client:
            response = await client.get(f"{base_url.rstrip('/')}/api/tags")
        if response.status_code != 200:
            raise unreachable
        pulled = {m["name"] for m in response.json()["models"]}
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        raise unreachable from exc
    missing = [m for m in required_models if m not in pulled]
    if missing:
        raise ConvoprobeError(
            "\n".join(f'Ollama model "{m}" is not pulled. Run: ollama pull {m}' for m in missing)
        )
