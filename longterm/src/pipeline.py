"""
Long-term pipeline orchestrator.

Run from this directory:
    python pipeline.py                  # full pipeline
    python pipeline.py --from-clean     # skip ingestion
    python pipeline.py --from-align     # skip ingestion + cleaning
    python pipeline.py --from-features  # only features + validation
    python pipeline.py --validate-only  # only validation
"""

from __future__ import annotations

import argparse
import logging
import sys
import time


def main():
    p = argparse.ArgumentParser(description="Long-term Frigg pipeline")
    p.add_argument("--from-clean",    action="store_true", help="skip ingestion")
    p.add_argument("--from-align",    action="store_true", help="skip ingestion + cleaning")
    p.add_argument("--from-features", action="store_true", help="run features + validation only")
    p.add_argument("--validate-only", action="store_true", help="only run validation")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("pipeline")

    t0 = time.time()

    # Imports here so config-driven dir-creation runs before sub-modules import
    if args.validate_only:
        import validation; validation.validate()
        log.info("Done in %.1fs", time.time() - t0)
        return

    if args.from_features:
        import features;   features.build_features()
        import validation; validation.validate()
        log.info("Done in %.1fs", time.time() - t0)
        return

    if args.from_align:
        import alignment;  alignment.align_all()
        import features;   features.build_features()
        import validation; validation.validate()
        log.info("Done in %.1fs", time.time() - t0)
        return

    if args.from_clean:
        import cleaning;   cleaning.clean_all()
        import alignment;  alignment.align_all()
        import features;   features.build_features()
        import validation; validation.validate()
        log.info("Done in %.1fs", time.time() - t0)
        return

    # Full pipeline
    import ingestion;  ingestion.ingest_all()
    import cleaning;   cleaning.clean_all()
    import alignment;  alignment.align_all()
    import features;   features.build_features()
    import validation; validation.validate()
    log.info("Done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    sys.exit(main())
