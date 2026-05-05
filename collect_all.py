"""
Master Dataset Collector — Kyrgyz Fake News Detection
=======================================================
Runs all source scrapers and merges them into a unified dataset.
Can also merge pre-existing CSVs without re-scraping.

Usage:
  # Full run (scrapes everything fresh):
  python collect_all.py

  # Merge only existing CSVs (no scraping):
  python collect_all.py --merge-only

  # Scrape specific sources only:
  python collect_all.py --sources 24kg azattyk factcheck

Final output: kyrgyz_fake_news_dataset.csv
"""

import argparse
import logging
import os
import pandas as pd
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [master] %(levelname)s %(message)s",
    handlers=[logging.FileHandler("collector_master.log"), logging.StreamHandler()],
)
log = logging.getLogger("master")

# ── Column schema — every source must conform to this ────────────────────────
SCHEMA = [
    "id",
    "headline",
    "body_text",
    "source",
    "url",
    "date",
    "language",
    "label",            # 0=real, 1=fake
    "label_confidence",
    "evidence_url",
    "source_type",
]

# ── Source map ────────────────────────────────────────────────────────────────
SOURCES = {
    "24kg": {
        "module":   "scraper_24kg",
        "out_csv":  "24kg_kyrgyz_data.csv",
        "label":    0,
    },
    "azattyk": {
        "module":   "scraper_azattyk",
        "out_csv":  "azattyk_kyrgyz_data.csv",
        "label":    0,
    },
    "sputnik": {
        "module":   "scraper_sputnik",
        "out_csv":  "sputnik_kyrgyz_data.csv",
        "label":    0,
    },
    "kaktus": {
        "module":   "scraper_kaktus",
        "out_csv":  "kaktus_kyrgyz_data.csv",
        "label":    0,
    },
    "saat": {
        "module": "scraper_saat",
        "out_csv": "saat_kyrgyz_data.csv",
        "label": 0,
    },
    "factcheck": {
        "module":   "scraper_factcheck",
        "out_csv":  "factcheck_fake_news.csv",
        "label":    1,
    },
}


# ── Normalise a DataFrame to the unified schema ───────────────────────────────
def normalise(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Add missing columns and standardise values."""
    cfg = SOURCES.get(source_name, {})

    # Ensure all schema columns exist
    for col in SCHEMA:
        if col not in df.columns:
            df[col] = None

    # Back-fill label from config if missing
    if df["label"].isna().all():
        df["label"] = cfg.get("label", -1)

    # Ensure IDs are unique
    df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]

    # Add language default
    if df["language"].isna().all():
        df["language"] = "kyrgyz"

    # evidence_url only matters for fake news
    if "evidence_url" not in df.columns or df["evidence_url"].isna().all():
        if cfg.get("label") == 1:
            df["evidence_url"] = df.get("url", None)
        else:
            df["evidence_url"] = None

    # Drop rows with no body text
    df = df[df["body_text"].notna() & (df["body_text"].str.len() >= 50)]

    return df[SCHEMA]


# ── Run a scraper module ──────────────────────────────────────────────────────
def run_scraper(source_name: str) -> pd.DataFrame | None:
    """Dynamically import and run a scraper module."""
    cfg = SOURCES[source_name]
    module_name = cfg["module"]

    try:
        import importlib
        mod = importlib.import_module(module_name)
        log.info(f"Running scraper: {module_name}")
        df = mod.main()
        return df
    except ImportError as exc:
        log.error(f"Could not import {module_name}: {exc}")
        return None
    except Exception as exc:
        log.error(f"Scraper {module_name} failed: {exc}")
        return None


# ── Load from existing CSV ────────────────────────────────────────────────────
def load_csv(path: str) -> pd.DataFrame | None:
    """Load and return a CSV if it exists."""
    if os.path.exists(path):
        df = pd.read_csv(path, encoding="utf-8-sig")
        log.info(f"  Loaded {len(df)} rows from {path}")
        return df
    log.warning(f"  File not found: {path}")
    return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main(sources_to_run: list[str] | None = None, merge_only: bool = False):
    log.info("=" * 70)
    log.info("KYRGYZ FAKE NEWS DATASET COLLECTOR")
    log.info("=" * 70)

    if sources_to_run is None:
        sources_to_run = list(SOURCES.keys())

    all_frames: list[pd.DataFrame] = []

    for name in sources_to_run:
        if name not in SOURCES:
            log.warning(f"Unknown source: {name}")
            continue

        cfg    = SOURCES[name]
        out_csv = cfg["out_csv"]
        df = None

        if not merge_only:
            # Try to scrape fresh
            df = run_scraper(name)

        if df is None:
            # Fall back to existing CSV
            log.info(f"Loading existing CSV for {name}...")
            df = load_csv(out_csv)

        if df is not None and not df.empty:
            df = normalise(df, name)
            all_frames.append(df)
            log.info(f"  {name}: {len(df)} rows (label={cfg['label']})")
        else:
            log.warning(f"  {name}: No data available — skipping.")

    if not all_frames:
        log.error("No data collected from any source.")
        return

    # ── Combine ───────────────────────────────────────────────────────────────
    combined = pd.concat(all_frames, ignore_index=True)

    # Remove duplicates by headline
    before = len(combined)
    combined = combined.drop_duplicates(subset=["headline"], keep="first")
    combined = combined.drop_duplicates(subset=["body_text"], keep="first")
    after  = len(combined)
    log.info(f"Deduplication: {before} → {after} rows ({before - after} removed)")

    # Remove rows with uncertain label
    combined = combined[combined["label"].isin([0, 1])]

    # Shuffle
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    out = "kyrgyz_fake_news_dataset.csv"
    combined.to_csv(out, index=False, encoding="utf-8-sig")
    log.info(f"\n✓ Final dataset saved → {out}")

    # ── Statistics ────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("DATASET STATISTICS")
    log.info("=" * 70)
    log.info(f"Total articles:    {len(combined)}")
    log.info(f"  Real news (0):   {(combined['label'] == 0).sum()}")
    log.info(f"  Fake news (1):   {(combined['label'] == 1).sum()}")
    log.info(f"\nBy source:")
    for src, cnt in combined["source"].value_counts().items():
        log.info(f"  {src}: {cnt}")
    log.info(f"\nBy language:")
    for lang, cnt in combined["language"].value_counts().items():
        log.info(f"  {lang}: {cnt}")

    avg_body = combined["body_text"].str.len().mean()
    log.info(f"\nAvg body_text length: {avg_body:.0f} chars")

    # Class balance
    real = (combined["label"] == 0).sum()
    fake = (combined["label"] == 1).sum()
    if fake > 0:
        ratio = real / fake
        log.info(f"Real:Fake ratio: {ratio:.2f}:1")

    log.info("=" * 70)
    return combined


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect and merge Kyrgyz fake news dataset")
    parser.add_argument(
        "--sources", nargs="+",
        choices=list(SOURCES.keys()),
        default=None,
        help="Which sources to include (default: all)",
    )
    parser.add_argument(
        "--merge-only", action="store_true",
        help="Skip scraping; just merge existing CSVs",
    )
    args = parser.parse_args()
    main(sources_to_run=args.sources, merge_only=args.merge_only)
