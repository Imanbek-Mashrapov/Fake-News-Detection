"""
Master Dataset Collector — Kyrgyz Fake News Detection  ·  v2
=============================================================
Runs all source scrapers and merges them into a unified dataset.

Pipeline steps:
  1. Scrape all real-news sources  (label=0)
  2. Scrape factcheck.kg + augment  (label=1)
  3. Detect & filter out Russian-language articles
  4. Deduplicate by headline + body_text
  5. Undersample real news so that Real:Fake ratio = 2:1
  6. Shuffle and save

Target totals (≈1000 rows, 2:1 balance):
  Real (label=0):  ~667 rows  ← undersample from larger pool
  Fake (label=1):  ~333 rows  ← from factcheck + augmentation

Usage:
  python collect_all.py                      # full run
  python collect_all.py --merge-only         # skip scraping, merge CSVs
  python collect_all.py --sources 24kg azattyk factcheck
"""

import argparse
import logging
import os
import importlib
import uuid

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [master] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("logs/collector_master.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("master")

# ── Column schema ─────────────────────────────────────────────────────────────
SCHEMA = [
    "id",
    "headline",
    "body_text",
    "source",
    "url",
    "date",
    "language",
    "label",             # 0=real, 1=fake
    "label_confidence",
    "evidence_url",
    "source_type",
]

# ── Source registry ───────────────────────────────────────────────────────────
SOURCES = {
    "24kg": {
        "module":  "source_scripts.scraper_24kg",
        "out_csv": "data/24kg_kyrgyz_data.csv",
        "label":   0,
        "target":  200,    # increased from 150
    },
    "azattyk": {
        "module":  "source_scripts.scraper_azattyk",
        "out_csv": "data/azattyk_kyrgyz_data.csv",
        "label":   0,
        "target":  200,
    },
    "sputnik": {
        "module":  "source_scripts.scraper_sputnik",
        "out_csv": "data/sputnik_kyrgyz_data.csv",
        "label":   0,
        "target":  200,
    },
    "kaktus": {
        "module":  "source_scripts.scraper_kaktus",
        "out_csv": "data/kaktus_kyrgyz_data.csv",
        "label":   0,
        "target":  150,
    },
    "saat": {
        "module":  "source_scripts.scraper_saat",
        "out_csv": "data/saat_kyrgyz_data.csv",
        "label":   0,
        "target":  200,
    },
    # ── NEW source ────────────────────────────────────────────────────────────
    "vestikg": {
        "module":  "source_scripts.scraper_vestikg",
        "out_csv": "data/vestikg_kyrgyz_data.csv",
        "label":   0,
        "target":  150,
    },
    # ── Fake news ─────────────────────────────────────────────────────────────
    "factcheck": {
        "module":  "source_scripts.scraper_factcheck",
        "out_csv": "data/factcheck_fake_news.csv",
        "label":   1,
        "target":  333,    # post-augmentation target
    },
}

# ── 2:1 balance targets ───────────────────────────────────────────────────────
TARGET_FAKE = 333
TARGET_REAL = TARGET_FAKE * 2   # 666
TOTAL_TARGET = TARGET_REAL + TARGET_FAKE  # 999


# ══════════════════════════════════════════════════════════════════════════════
# Language filtering
# ══════════════════════════════════════════════════════════════════════════════

def detect_language(text: str) -> str:
    """
    Detect whether text is primarily Kyrgyz or Russian.
    Uses Kyrgyz-specific Unicode characters (Ө Ү Ң Ғ) as a signal.
    """
    kyrgyz_specific = set("өүңғ")
    count = sum(1 for c in str(text).lower() if c in kyrgyz_specific)
    return "kyrgyz" if count >= 5 else "russian"


def filter_kyrgyz(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where headline + body_text are not primarily Kyrgyz."""
    before = len(df)

    def is_kyrgyz_row(row) -> bool:
        combined = str(row.get("headline", "")) + " " + str(row.get("body_text", ""))
        return detect_language(combined) == "kyrgyz"

    mask = df.apply(is_kyrgyz_row, axis=1)
    df_filtered = df[mask].copy()
    log.info(f"Language filter: {before} → {len(df_filtered)} rows "
             f"({before - len(df_filtered)} non-Kyrgyz removed)")
    return df_filtered


# ══════════════════════════════════════════════════════════════════════════════
# Schema normalisation
# ══════════════════════════════════════════════════════════════════════════════

def normalise(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Ensure every row conforms to SCHEMA; add missing columns."""
    cfg = SOURCES.get(source_name, {})

    for col in SCHEMA:
        if col not in df.columns:
            df[col] = None

    # Back-fill label from config if missing
    if df["label"].isna().all():
        df["label"] = cfg.get("label", -1)

    # Unique IDs
    df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]

    if df["language"].isna().all():
        df["language"] = "kyrgyz"

    if cfg.get("label") == 1 and (df["evidence_url"].isna().all()):
        df["evidence_url"] = df.get("url", None)
    elif cfg.get("label") == 0:
        df["evidence_url"] = None

    # Drop very short articles
    df = df[df["body_text"].notna() & (df["body_text"].str.len() >= 80)]

    return df[SCHEMA]


# ══════════════════════════════════════════════════════════════════════════════
# Scraper runner & CSV loader
# ══════════════════════════════════════════════════════════════════════════════

def run_scraper(source_name: str) -> pd.DataFrame | None:
    cfg = SOURCES[source_name]
    module_name = cfg["module"]
    try:
        mod = importlib.import_module(module_name)
        log.info(f"Running scraper: {module_name}")
        return mod.main()
    except ImportError as exc:
        log.error(f"Could not import {module_name}: {exc}")
        return None
    except Exception as exc:
        log.error(f"Scraper {module_name} failed: {exc}")
        return None


def load_csv(path: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        df = pd.read_csv(path, encoding="utf-8-sig")
        log.info(f"  Loaded {len(df)} rows from {path}")
        return df
    log.warning(f"  File not found: {path}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(sources_to_run: list[str] | None = None, merge_only: bool = False):
    log.info("=" * 70)
    log.info("KYRGYZ FAKE NEWS DATASET COLLECTOR  v2")
    log.info("=" * 70)

    if sources_to_run is None:
        sources_to_run = list(SOURCES.keys())

    all_frames: list[pd.DataFrame] = []

    # ── Step 1: Collect from each source ──────────────────────────────────────
    for name in sources_to_run:
        if name not in SOURCES:
            log.warning(f"Unknown source: {name}")
            continue

        cfg     = SOURCES[name]
        out_csv = cfg["out_csv"]
        df      = None

        if not merge_only:
            df = run_scraper(name)

        if df is None:
            log.info(f"Loading existing CSV for '{name}'…")
            df = load_csv(out_csv)

        if df is not None and not df.empty:
            df = normalise(df, name)
            all_frames.append(df)
            log.info(f"  {name}: {len(df)} rows after normalisation (label={cfg['label']})")
        else:
            log.warning(f"  {name}: No data — skipping.")

    if not all_frames:
        log.error("No data collected from any source.")
        return

    combined = pd.concat(all_frames, ignore_index=True)

    # ── Step 2: Deduplicate ───────────────────────────────────────────────────
    before = len(combined)
    combined = combined.drop_duplicates(subset=["headline"], keep="first")
    combined = combined.drop_duplicates(subset=["body_text"],  keep="first")
    log.info(f"Deduplication: {before} → {len(combined)} rows ({before - len(combined)} removed)")

    # ── Step 3: Language filtering — keep only Kyrgyz ─────────────────────────
    combined = filter_kyrgyz(combined)

    # ── Step 4: Keep only clean labels ────────────────────────────────────────
    combined = combined[combined["label"].isin([0, 1])]
    log.info(f"After label filter: {len(combined)} rows")

    # ── Step 5: 2:1 undersampling ─────────────────────────────────────────────
    df_real = combined[combined["label"] == 0]
    df_fake = combined[combined["label"] == 1]

    log.info(f"Available: {len(df_real)} real,  {len(df_fake)} fake")

    # Cap fake at TARGET_FAKE
    if len(df_fake) > TARGET_FAKE:
        df_fake = df_fake.sample(n=TARGET_FAKE, random_state=42)
        log.info(f"Fake capped at {TARGET_FAKE}")

    actual_fake = len(df_fake)
    actual_real_target = actual_fake * 2

    # Undersample real if we have more than needed
    if len(df_real) > actual_real_target:
        df_real = df_real.sample(n=actual_real_target, random_state=42)
        log.info(f"Real undersampled to {actual_real_target} (2×{actual_fake})")
    else:
        log.warning(
            f"Only {len(df_real)} real articles available "
            f"(wanted {actual_real_target}) — ratio will be sub-optimal."
        )

    combined = pd.concat([df_real, df_fake], ignore_index=True)

    # ── Step 6: Shuffle & save ────────────────────────────────────────────────
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    out = "data/kyrgyz_fake_news_dataset.csv"
    combined.to_csv(out, index=False, encoding="utf-8-sig")
    log.info(f"\n✓ Final dataset saved → {out}")

    # ── Step 7: Statistics ────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("DATASET STATISTICS")
    log.info("=" * 70)
    log.info(f"Total articles:   {len(combined)}")
    log.info(f"  Real (0):       {(combined['label'] == 0).sum()}")
    log.info(f"  Fake (1):       {(combined['label'] == 1).sum()}")

    real_cnt = (combined["label"] == 0).sum()
    fake_cnt = (combined["label"] == 1).sum()
    if fake_cnt > 0:
        log.info(f"  Real:Fake ratio: {real_cnt/fake_cnt:.2f}:1")

    log.info("\nBy source:")
    for src, cnt in combined["source"].value_counts().items():
        log.info(f"  {src}: {cnt}")

    log.info("\nBy language:")
    for lang, cnt in combined["language"].value_counts().items():
        log.info(f"  {lang}: {cnt}")

    avg_body = combined["body_text"].str.len().mean()
    log.info(f"\nAvg body_text length: {avg_body:.0f} chars")
    log.info("=" * 70)

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect & merge Kyrgyz fake news dataset (v2)")
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