#!/usr/bin/env python3
"""
Migration: populate audio_segments table from existing filesystem data.

Scans app/data/preprocessed/collection/<date>/<call_id>/*_seg*.wav and
inserts rows into audio_segments.  Safe to run multiple times (deletes
existing rows first for the same recording).

Usage:
    python -m train.migrate_segments
    python -m train.migrate_segments --dry-run   # just report what would happen
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path
from typing import List, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("migrate_segments")

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent  # app/train/
PROJECT_ROOT = HERE.parent               # app/
DB_PATH = PROJECT_ROOT / "data" / "training.db"
PREPROCESSED = PROJECT_ROOT / "data" / "preprocessed" / "collection"


def parse_seg_index(seg_path: Path) -> int:
    """Extract segment index from a path like .../seg_3.wav or .../seg_12.wav."""
    name = seg_path.stem  # e.g. "075582095333_agent_0" or "seg_3"
    # Try to extract trailing number
    import re
    nums = re.findall(r"_(\d+)$", name)
    if nums:
        return int(nums[-1])
    # fallback: sort by name alphabetically
    return 0


def scan_existing_segments() -> List[Tuple[str, str, int, float, float, int]]:
    """
    Scan preprocessed directory and return list of:
      (call_id, file_path_rel, segment_index, start_sec, end_sec, recording_id)
    
    file_path_rel is relative to PROJECT_ROOT.
    """
    if not PREPROCESSED.exists():
        logger.warning("Preprocessed directory not found: %s", PREPROCESSED)
        return []

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Build call_id -> recording_id map
    call_to_rec = {}
    rows = conn.execute("SELECT id, call_id FROM recordings").fetchall()
    for r in rows:
        call_to_rec[r["call_id"]] = r["id"]

    results: List[Tuple[str, str, int, float, float, int]] = []
    found = 0
    missing = 0

    for date_dir in sorted(PREPROCESSED.iterdir()):
        if not date_dir.is_dir():
            continue
        for call_dir in sorted(date_dir.iterdir()):
            call_id = call_dir.name
            rec_id = call_to_rec.get(call_id)
            if rec_id is None:
                logger.warning("  ⚠️  Call %s not in DB, skipping", call_id)
                missing += 1
                continue

            seg_files = sorted(call_dir.glob("*_seg*.wav"))
            for idx, seg_path in enumerate(seg_files):
                rel_path = str(seg_path.relative_to(PROJECT_ROOT))
                # Estimate start/end from filename if available, else 0
                start = 0.0
                end = 0.0
                # Try to parse from seg_N pattern
                seg_idx = parse_seg_index(seg_path)
                results.append((call_id, rel_path, seg_idx, start, end, rec_id))

            found += len(seg_files)

    conn.close()
    logger.info("Scanned: %d segments for %d calls (calls not in DB: %d)",
                found, len(results), missing)
    return results


def migrate(dry_run: bool = False) -> int:
    """Populate audio_segments table. Returns number of rows inserted."""
    segments = scan_existing_segments()
    if not segments:
        logger.info("Nothing to migrate.")
        return 0

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # Delete existing entries for affected recordings (safe re-run)
    rec_ids = set(s[5] for s in segments)  # recording_id at index 5
    existing_count = conn.execute(
        f"SELECT COUNT(*) FROM audio_segments WHERE recording_id IN ({','.join('?' for _ in rec_ids)})",
        list(rec_ids),
    ).fetchone()[0]
    if existing_count > 0:
        if dry_run:
            logger.info("  Would delete %d existing segment rows for %d recordings",
                        existing_count, len(rec_ids))
        else:
            conn.execute(
                f"DELETE FROM audio_segments WHERE recording_id IN ({','.join('?' for _ in rec_ids)})",
                list(rec_ids),
            )
            logger.info("  Deleted %d existing segment rows", existing_count)

    # Build insert data grouped by recording_id
    by_rec: dict = {}
    for call_id, rel_path, seg_idx, start, end, rec_id in segments:
        if rec_id not in by_rec:
            by_rec[rec_id] = {"call_id": call_id, "segments": []}
        by_rec[rec_id]["segments"].append((rel_path, seg_idx, start, end))

    insert_sql = """INSERT INTO audio_segments
        (recording_id, segment_index, file_path, start_sec, end_sec, speaker_label, speaker_type, label_source)
        VALUES (?, ?, ?, ?, ?, '', 'unknown', 'migrated')"""

    total = 0
    for rec_id, data in by_rec.items():
        rows_to_insert = []
        for rel_path, seg_idx, start, end in sorted(data["segments"], key=lambda x: x[1]):
            rows_to_insert.append((rec_id, seg_idx, rel_path, start, end))

        if dry_run:
            logger.info("  [dry-run] Would insert %d segments for rec_id=%s (%s)",
                        len(rows_to_insert), rec_id, data["call_id"])
        else:
            conn.executemany(insert_sql, rows_to_insert)
            total += len(rows_to_insert)

    if not dry_run:
        conn.commit()
    conn.close()

    if dry_run:
        logger.info("Dry-run complete. Would insert %d total rows across %d recordings.",
                    total, len(by_rec))
    else:
        logger.info("Migrated %d segment rows across %d recordings to audio_segments table.",
                    total, len(by_rec))
    return total


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    migrate(dry_run=dry_run)


if __name__ == "__main__":
    main()
