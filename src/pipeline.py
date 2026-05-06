"""
Pipeline orchestrator.

Runs all stages in order:
  1. Ingestion  → data/raw/
  2. Cleaning   → data/clean/
  3. Alignment  → data/aligned/base_dataset.parquet
  4. Features   → data/processed/final_dataset.parquet
  5. Validation → pass/fail

Usage:
  python pipeline.py              # full pipeline
  python pipeline.py --from-clean # skip ingestion (raw files exist)
  python pipeline.py --from-align # skip ingestion + cleaning
  python pipeline.py --validate-only
"""

from __future__ import annotations

import sys
import os
import logging
import argparse
from pathlib import Path

# Load .env from repo root if present (so teammates don't need to export manually)
_env_file = Path(__file__).parent.parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        if _line.strip() and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run(skip_ingest: bool = False, skip_clean: bool = False, validate_only: bool = False):
    from config import TRAIN_START, TRAIN_END

    if not validate_only:
        if not skip_ingest and not skip_clean:
            log.info("━" * 60)
            log.info("STAGE 1 — Ingestion")
            log.info("━" * 60)
            from ingestion import ingest_all
            ingest_all(TRAIN_START, TRAIN_END)

        if not skip_clean:
            log.info("━" * 60)
            log.info("STAGE 2 — Cleaning")
            log.info("━" * 60)
            from cleaning import clean_all
            clean_all()

        log.info("━" * 60)
        log.info("STAGE 3 — Alignment")
        log.info("━" * 60)
        from alignment import align_all
        align_all()

        log.info("━" * 60)
        log.info("STAGE 4 — Feature engineering")
        log.info("━" * 60)
        from features import engineer_features
        engineer_features()

    log.info("━" * 60)
    log.info("STAGE 5 — Validation")
    log.info("━" * 60)
    from validation import validate
    validate()

    log.info("━" * 60)
    log.info("Pipeline complete. Final dataset at data/processed/final_dataset.parquet")
    log.info("━" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-clean",    action="store_true", help="Skip ingestion")
    parser.add_argument("--from-align",    action="store_true", help="Skip ingestion + cleaning")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation")
    args = parser.parse_args()

    run(
        skip_ingest=args.from_clean or args.from_align,
        skip_clean=args.from_align,
        validate_only=args.validate_only,
    )
