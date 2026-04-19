"""
Runtime verification helper for LLM connectivity and model-id correctness.

What it checks:
  - LLM server is reachable via OpenAI-compat `/v1/models`
  - `settings.llm_model_name` exists in the returned model ids
  - If `settings.vision_model` is set, it also exists in model ids

Why this exists:
  - With OpenAI-compat backends (llama.cpp, vLLM), the "model id" must match
    what the server returns from `/v1/models`. A mismatch leads to confusing
    404/400 errors at runtime.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _extract_model_ids(payload: dict[str, Any]) -> list[str]:
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    ids: list[str] = []
    for item in data:
        if isinstance(item, dict):
            mid = item.get("id")
            if isinstance(mid, str) and mid.strip():
                ids.append(mid.strip())
    return ids


def _read_env_file(path: str = ".env") -> dict[str, str]:
    """
    Minimal .env parser (KEY=VALUE, supports comments and blank lines).
    Does not support complex quoting/escaping; good enough for our config keys.
    """
    vals: dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k:
                    vals[k] = v
    except FileNotFoundError:
        pass
    return vals


def _first_nonempty(values: Iterable[str | None]) -> str:
    for v in values:
        if v is None:
            continue
        vv = str(v).strip()
        if vv:
            return vv
    return ""


def main() -> int:
    file_env = _read_env_file(".env")

    llm_backend = _first_nonempty([
        os.getenv("LLM_BACKEND"),
        file_env.get("LLM_BACKEND"),
        "llama.cpp",
    ])
    llm_server_url = _first_nonempty([
        os.getenv("LLM_SERVER_URL"),
        file_env.get("LLM_SERVER_URL"),
        os.getenv("VLLM_SERVER_URL"),
        file_env.get("VLLM_SERVER_URL"),
    ])
    llm_model_name = _first_nonempty([
        os.getenv("LLM_MODEL_NAME"),
        file_env.get("LLM_MODEL_NAME"),
        os.getenv("VLLM_MODEL_NAME"),
        file_env.get("VLLM_MODEL_NAME"),
    ])
    vision_model = _first_nonempty([
        os.getenv("VISION_MODEL"),
        file_env.get("VISION_MODEL"),
    ])

    base = (llm_server_url or "").rstrip("/")
    if not base:
        print("ERROR: LLM server URL is empty. Set LLM_SERVER_URL (or VLLM_SERVER_URL).")
        return 2

    models_url = f"{base}/models"
    print("LLM runtime check")
    print(f"- llm_backend     : {llm_backend!r}")
    print(f"- llm_server_url  : {base!r}")
    print(f"- llm_model_name  : {llm_model_name!r}")
    print(f"- vision_model    : {vision_model!r}")
    print(f"- GET {models_url}")
    if llm_backend.lower().startswith("llama") and ":8000" in base:
        print("WARN: llm_backend is llama.cpp but URL looks like a vLLM default (port 8000).")
        print("      If you switched to llama-server, set LLM_SERVER_URL=http://localhost:8080/v1")

    try:
        req = Request(
            models_url,
            headers={
                "Accept": "application/json",
                # Some OpenAI-compat servers tolerate missing auth, but clients may send it.
                "Authorization": "Bearer dummy",
            },
            method="GET",
        )
        with urlopen(req, timeout=8.0) as resp:
            raw = resp.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"ERROR: cannot reach LLM server models endpoint: {exc}")
        return 3

    try:
        payload = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        print("ERROR: /models did not return JSON.")
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            text = repr(raw)
        print(text[:4000])
        return 4

    model_ids = _extract_model_ids(payload)
    if not model_ids:
        print("ERROR: /models returned no model ids.")
        print(json.dumps(payload, ensure_ascii=False)[:4000])
        return 5

    ok = True
    if llm_model_name and llm_model_name not in model_ids:
        ok = False
        print("ERROR: LLM_MODEL_NAME does not match any server model id.")
        print("  - Hint: run `curl <LLM_SERVER_URL>/models` and copy the exact `id`.")

    if (vision_model or "").strip():
        if vision_model not in model_ids:
            ok = False
            print("ERROR: VISION_MODEL is set but does not match any server model id.")

    print(f"OK: server returned {len(model_ids)} model id(s).")
    for mid in model_ids[:20]:
        print(f"  - {mid}")
    if len(model_ids) > 20:
        print(f"  ... (+{len(model_ids) - 20} more)")

    return 0 if ok else 6


if __name__ == "__main__":
    raise SystemExit(main())

