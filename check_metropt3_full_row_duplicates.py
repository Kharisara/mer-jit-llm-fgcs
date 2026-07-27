#!/usr/bin/env python3
"""Audit byte-normalized duplicate data rows in the canonical MetroPT-3 CSV.

The check streams the CSV in binary mode, excludes the header, removes only the
line terminator (CR/LF), hashes each complete row with SHA-256, and stores the
32-byte digests in a temporary SQLite database with a UNIQUE primary key. This
provides a deterministic full-row duplicate count without loading the CSV into
memory.

Optionally updates the existing preparation manifest with the audit result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_FILENAME = "MetroPT3(AirCompressor).csv"
EXPECTED_ROWS = 1_516_948
EXPECTED_SHA256 = "db30ccb4ea402e3c8bf2c99db06e288d4f2a772f6928f9dbe26a920d69793e24"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-csv", required=True, type=Path)
    p.add_argument("--output-json", required=True, type=Path)
    p.add_argument("--manifest-json", type=Path,
                   help="Existing preparation manifest to update in place")
    p.add_argument("--commit-every", type=int, default=50_000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source_csv
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.name != EXPECTED_FILENAME:
        raise RuntimeError(f"Unexpected filename: {source.name!r}")
    source_sha = sha256_file(source)
    if source_sha != EXPECTED_SHA256:
        raise RuntimeError("Source SHA-256 does not match the canonical UCI file")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="metropt_dup_audit_") as td:
        db = Path(td) / "digests.sqlite3"
        conn = sqlite3.connect(db)
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("CREATE TABLE rows (digest BLOB PRIMARY KEY)")

        total = unique = duplicates = 0
        first_duplicate_data_row = None
        with source.open("rb") as f:
            header = f.readline()
            if not header:
                raise RuntimeError("Source CSV is empty")
            for data_row_number, raw in enumerate(f, start=1):
                row = raw.rstrip(b"\r\n")
                digest = hashlib.sha256(row).digest()
                try:
                    conn.execute("INSERT INTO rows(digest) VALUES (?)", (digest,))
                    unique += 1
                except sqlite3.IntegrityError:
                    duplicates += 1
                    if first_duplicate_data_row is None:
                        first_duplicate_data_row = data_row_number
                total += 1
                if total % args.commit_every == 0:
                    conn.commit()
                    print(f"[INFO] rows={total:,} unique={unique:,} duplicates={duplicates:,}")
        conn.commit()
        conn.close()

    if total != EXPECTED_ROWS:
        raise RuntimeError(f"Observed {total:,} data rows; expected {EXPECTED_ROWS:,}")

    result = {
        "dataset": "MetroPT-3",
        "source_filename": source.name,
        "source_sha256": source_sha,
        "source_rows": total,
        "duplicate_check_scope": "complete canonical source data rows excluding header",
        "duplicate_check_normalization": "remove CR/LF terminator only; preserve all other row bytes",
        "duplicate_check_digest": "SHA-256 per complete row stored under SQLite UNIQUE constraint",
        "unique_full_rows": unique,
        "duplicate_full_rows": duplicates,
        "first_duplicate_data_row": first_duplicate_data_row,
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.manifest_json:
        manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))
        manifest.update({
            "source_duplicate_row_check_scope": result["duplicate_check_scope"],
            "source_duplicate_row_check_normalization": result["duplicate_check_normalization"],
            "source_duplicate_row_check_method": result["duplicate_check_digest"],
            "source_unique_full_rows": unique,
            "source_duplicate_full_rows": duplicates,
            "source_first_duplicate_data_row": first_duplicate_data_row,
            "source_duplicate_row_audit_json": str(args.output_json),
        })
        args.manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[OUT] Updated manifest: {args.manifest_json}")

    print(f"[OUT] {args.output_json}")
    print(f"[PASS] rows={total:,} unique={unique:,} duplicate_rows={duplicates:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
