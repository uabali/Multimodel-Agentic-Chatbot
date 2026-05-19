#!/usr/bin/env python3
"""Bulk-index supported files from a directory into Qdrant.

Usage:
  uv run python scripts/index_corpus.py
  uv run python scripts/index_corpus.py --dir data/corpus --smart-reindex
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.ingest import index_directory  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Index a local document directory into Qdrant.")
    parser.add_argument(
        "--dir",
        default="data",
        help="Directory containing documents to index (default: data)",
    )
    parser.add_argument(
        "--smart-reindex",
        action="store_true",
        help="Replace collection via fingerprint-based smart_reindex instead of per-file ingest",
    )
    args = parser.parse_args()

    results = index_directory(args.dir, smart_reindex=args.smart_reindex)
    if not results:
        logger.error("No files indexed from %s", args.dir)
        return 1

    ok = sum(1 for r in results if r.get("status") in {"success", "reindexed", "skipped"})
    logger.info("Indexed %d/%d entries", ok, len(results))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
