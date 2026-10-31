#!/usr/bin/env python3
"""Validate cloud-run summary and export only non-row-level evidence.

Never publishes: MELD provider source, canonical two-column input, replay trace
rows, receipts, event-level ground-truth labels, individual replay-point IDs,
or raw ZIPs made by array_r1_cloud_full_model.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

EXPECTED_INPUT_SHA256 = '2b46fd5e6887305d2eb43b1f4383e23ed3c1062a4826e1b7a428a10bd8f84f44'

# Allowlisted, aggregate-only outputs; keep this restrictive.
ALLOWLIST = (
    'run_summary.json',
    'environment_and_provenance.json',
    'file_integrity_manifest.json',
    'pilot_validator_results.json',
    'cloud_comparison_by_mode.csv',
    'cloud_comparison_by_policy.csv',
    'fresh_executions/execution_integrity_manifest.json',
    'fresh_executions/execution_receipt_validation_summary.csv',
    'fresh_executions/record_bound_corruption_validation_summary.csv',
    'label_excluded_validation/baseline_comparison_by_fault_class.csv',
    'label_excluded_validation/phase1_validation_manifest.json',
)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1_048_576), b''):
            digest.update(block)
    return digest.hexdigest()


def validate(source: Path) -> dict:
    summary = json.loads((source / 'run_summary.json').read_text(encoding='utf-8'))
    provenance = json.loads((source / 'environment_and_provenance.json').read_text(encoding='utf-8'))
    if summary.get('status') != 'passed':
        raise ValueError(f"Run did not pass: {summary.get('status')}")
    mode = summary.get('mode')
    if mode not in ('pilot', 'full'):
        raise ValueError(f'Unexpected run mode: {mode}')
    if provenance.get('input_csv_sha256') != EXPECTED_INPUT_SHA256:
        raise ValueError('Experiment used a different primary replay workload')
    if not provenance.get('git_head_seen_inside_environment') or provenance['git_head_seen_inside_environment'] == 'unavailable':
        raise ValueError('Cannot authenticate source Git commit')
    if provenance.get('source_commit_env') != provenance['git_head_seen_inside_environment']:
        raise ValueError('Git checkout revision disagrees with source commit environment')
    metrics = summary.get('metrics', {})
    if mode == 'pilot':
        rows = metrics.get('pilot_validator_results', [])
        if len(rows) != 5 or set(x['mode'] for x in rows) != {
            'clean', 'unlogged_downstream_call', 'false_execution_log',
            'duplicate_downstream_call', 'mismatched_correlation_id'
        }:
            raise ValueError('Pilot five-mode test is incomplete')
        for row in rows:
            if int(row['full']) != int(row['mode'] != 'clean'):
                raise ValueError(f'Pilot detection check failed: {row}')
    else:
        expected = {
            'fresh_receipt_execution_instances': 90,
            'fresh_clean_execution_instances': 18,
            'fresh_receipt_fault_execution_instances': 72,
            'label_excluded_evidence_units': 270,
            'positive_units': 228,
            'negative_units': 42,
            'full_positive_detections': 228,
            'full_false_alarms': 0,
            'localized_events_fp': 0,
            'localized_events_fn': 0,
        }
        for key, value in expected.items():
            if int(metrics.get(key, -1)) != value:
                raise ValueError(f'{key}: expected {value}, saw {metrics.get(key)}')
    # Confirm every raw file checksum made by the cloud wrapper is intact.
    files = json.loads((source / 'file_integrity_manifest.json').read_text(encoding='utf-8'))
    if not isinstance(files, list) or not files:
        raise ValueError('Missing raw file integrity provenance')
    for entry in files:
        rel = entry.get('path', '')
        path = (source / rel).resolve()
        if source.resolve() not in path.parents or not path.is_file():
            raise ValueError(f'Invalid or missing input provenance path: {rel!r}')
        if path.stat().st_size != int(entry.get('bytes', -1)):
            raise ValueError(f'File size does not match source manifest: {rel}')
        if sha256(path) != entry.get('sha256'):
            raise ValueError(f'File SHA-256 does not match source manifest: {rel}')
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True, type=Path)
    ap.add_argument('--destination', required=True, type=Path)
    ns = ap.parse_args()
    source = ns.source.resolve()
    destination = ns.destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Aggregate destination must be separate from raw source')
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Aggregate destination must be new/empty')
    summary = validate(source)
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for rel in ALLOWLIST:
        origin = source / rel
        if origin.is_file():
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)
            copied.append(rel)
    if not (destination / 'run_summary.json').is_file():
        raise RuntimeError('No summary copied')
    note = (
        '# ReplayBench-PG R1 GitHub-hosted validation evidence\n\n'
        f"Mode: **{summary['mode']}**. Source run passed the bundled verification checks.\n\n"
        'These files contain **aggregated results and hashes only**, not provider-obtained MELD '\
        'data, canonical derived inputs, event-level receipt or trace rows, or fault-target IDs.\n\n'
        'Runner: GitHub-hosted standard Ubuntu runner. Execution uses a deterministic in-process '\
        'downstream stub and receipt collector. This does **not** demonstrate independent '\
        'external-service attestation or real multi-service fault tolerance.\n\n'
        f'Included files: {len(copied)}.\n'
    )
    (destination / 'README.md').write_text(note, encoding='utf-8')
    receipt = {'mode': summary['mode'], 'files': copied, 'original_source_checks_passed': True,
               'source_sha256': {x: sha256(destination/x) for x in copied}}
    (destination / 'aggregate_export_manifest.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n',encoding='utf-8')
    with open(os.environ.get('GITHUB_STEP_SUMMARY', os.devnull), 'a', encoding='utf-8') as handle:
        handle.write(f"## ReplayBench-PG R1 {summary['mode']} passed\n\n")
        handle.write('Aggregate-only evidence has been uploaded; no MELD records or raw traces were uploaded.\n\n')
        if summary['mode'] == 'full':
            m = summary['metrics']
            handle.write(f"- Fresh receipt-enabled runs: **{m['fresh_receipt_execution_instances']}**\n")
            handle.write(f"- Label-independent evidence units: **{m['label_excluded_evidence_units']}**\n")
            handle.write(f"- Full validator: **{m['full_positive_detections']}/{m['positive_units']}** positive; **{m['full_false_alarms']}/{m['negative_units']}** false alarms\n")
    print('[PASS] Aggregate export audit and full raw-file checksum verification passed')
    print(f'[DONE] Aggregate-only export: {destination}, {len(copied)} allowlisted inputs')


if __name__ == '__main__':
    main()
