#!/usr/bin/env python3
"""
Benchmark llama.cpp vs MLX-LM OpenAI-compatible chat endpoints.

This is a gate, not a migration switch. Run both servers, then compare:

  uv run python scripts/benchmark_llm_backends.py

Environment:
  LLAMA_SERVER_URL   default: LLM_SERVER_URL or http://localhost:8080/v1
  LLAMA_MODEL_NAME   default: LLM_MODEL_NAME or gemma-4-e4b
  MLX_SERVER_URL     default: http://localhost:8081/v1
  MLX_MODEL_NAME     default: LLM_MODEL_NAME or gemma-4-e4b
  BENCH_MAX_TOKENS   default: 256
  VISION_IMAGE_PATH  optional local image for a vision compatibility probe
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import statistics
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROMPTS: list[tuple[str, str]] = [
    ("direct_chat", "Selam. Kendini tek cumlede tanit."),
    (
        "rag",
        "Baglam:\n"
        "[1] Frappe, yerel LLM, hibrit retrieval ve kaynakli yanit uretimi kullanan bir asistandir.\n"
        "[2] Gecikme hedefi ilk token suresini ve toplam yanit suresini dusurmektir.\n\n"
        "Soru: Bu sistemin amaci nedir? Kaynak numaralariyla kisa cevap ver.",
    ),
    (
        "web_like",
        "Bir web arama sonucundan cevap veriyormus gibi davran. Konu: Apple Silicon yerel LLM calistirma. "
        "Kisa, temkinli ve tarih belirtmeden genel cevap ver.",
    ),
    (
        "long_context",
        ("Asagidaki notlari ozetle:\n" + "Yerel RAG gecikmesini azaltmak icin router, cache, reranker ve stream TTFT olculur.\n" * 120),
    ),
]


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _post_json(url: str, payload: dict[str, Any], timeout: float = 120.0):
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": "Bearer dummy",
        },
        method="POST",
    )
    return urlopen(req, timeout=timeout)


def _iter_sse_lines(resp):
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        yield data


def _extract_delta_chars(payload: dict[str, Any]) -> str:
    try:
        choice = payload.get("choices", [{}])[0]
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str):
            return content
        message = choice.get("message") or {}
        content = message.get("content")
        return content if isinstance(content, str) else ""
    except Exception:
        return ""


def _run_case(base_url: str, model: str, name: str, prompt: Any, max_tokens: int) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
    }
    started = time.perf_counter()
    chars = 0
    ttft_ms: float | None = None
    try:
        with _post_json(url, payload) as resp:
            for data in _iter_sse_lines(resp):
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = _extract_delta_chars(chunk)
                if delta:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - started) * 1000
                    chars += len(delta)
    except (HTTPError, URLError, TimeoutError) as exc:
        return {"case": name, "ok": False, "error": str(exc)}
    total_ms = (time.perf_counter() - started) * 1000
    return {
        "case": name,
        "ok": True,
        "ttft_ms": round(ttft_ms or total_ms, 1),
        "total_ms": round(total_ms, 1),
        "chars": chars,
    }


def _vision_prompt(path: str) -> Any:
    p = Path(path)
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    encoded = base64.b64encode(p.read_bytes()).decode("ascii")
    return [
        {"type": "text", "text": "Bu gorseli tek cumlede acikla."},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
    ]


def _summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    ok_rows = [r for r in rows if r.get("ok")]
    if not ok_rows:
        return {}
    return {
        "median_ttft_ms": round(statistics.median(float(r["ttft_ms"]) for r in ok_rows), 1),
        "median_total_ms": round(statistics.median(float(r["total_ms"]) for r in ok_rows), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=int(_env("BENCH_MAX_TOKENS", "256")))
    args = parser.parse_args()

    endpoints = [
        ("llama.cpp", _env("LLAMA_SERVER_URL", _env("LLM_SERVER_URL", "http://localhost:8080/v1")), _env("LLAMA_MODEL_NAME", _env("LLM_MODEL_NAME", "gemma-4-e4b"))),
        ("mlx", _env("MLX_SERVER_URL", "http://localhost:8081/v1"), _env("MLX_MODEL_NAME", _env("LLM_MODEL_NAME", "gemma-4-e4b"))),
    ]
    prompt_cases = list(PROMPTS)
    if _env("VISION_IMAGE_PATH"):
        prompt_cases.append(("vision", _vision_prompt(_env("VISION_IMAGE_PATH"))))

    all_results: dict[str, list[dict[str, Any]]] = {}
    for backend, base_url, model in endpoints:
        print(f"\nbackend={backend} base_url={base_url} model={model}")
        rows: list[dict[str, Any]] = []
        for name, prompt in prompt_cases:
            row = _run_case(base_url, model, name, prompt, args.max_tokens)
            rows.append(row)
            if row.get("ok"):
                print(
                    f"case={name} ok=true ttft_ms={row['ttft_ms']} "
                    f"total_ms={row['total_ms']} chars={row['chars']}"
                )
            else:
                print(f"case={name} ok=false error={row.get('error')}")
        summary = _summary(rows)
        if summary:
            print(
                f"summary median_ttft_ms={summary['median_ttft_ms']} "
                f"median_total_ms={summary['median_total_ms']}"
            )
        all_results[backend] = rows

    print("\njson=" + json.dumps(all_results, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
