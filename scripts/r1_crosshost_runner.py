#!/usr/bin/env python3
"""Two-VM R1 transport experiment reusing the unchanged ReplayBench validator.

VM A hosts worker 0, VM B hosts worker 1, collector and coordinator.
The existing scenario generator/validator is reused, not reimplemented.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from scripts import r1_dist_runner as original

SECRET = os.environ.get('R1_CROSSHOST_TOKEN', '')
if len(SECRET) < 32:
    raise SystemExit('R1_CROSSHOST_TOKEN must be a random secret of at least 32 characters')


def request_json(base, path, data=None, timeout=4):
    body = None if data is None else json.dumps(data, separators=(',', ':')).encode('utf-8')
    headers = {'X-R1-Crosshost-Token': SECRET}
    if body is not None:
        headers['Content-Type'] = 'application/json'
    req = Request(base + path, data=body, headers=headers,
                  method='POST' if body is not None else 'GET')
    try:
        with urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


original.request_json = request_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--remote-host', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    remote_host = args.remote_host
    if not all(c.isalnum() or c == '-' for c in remote_host):
        raise SystemExit('Remote hostname must be a MagicDNS hostname, not a URL')
    original.WORKERS = (f'http://{remote_host}:18761', 'http://127.0.0.1:18762')
    original.COLLECTOR = 'http://127.0.0.1:18766'
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    # May wait for the other GitHub Actions job to provision its VM.
    health = {
        original.WORKERS[0]: original.wait_healthy(original.WORKERS[0], until=240),
        original.WORKERS[1]: original.wait_healthy(original.WORKERS[1], until=15),
        original.COLLECTOR: original.wait_healthy(original.COLLECTOR, until=15),
    }
    remote = health[original.WORKERS[0]]
    local = health[original.WORKERS[1]]
    collector = health[original.COLLECTOR]
    if remote['host_id'] == local['host_id']:
        raise SystemExit('NOT CROSS-HOST: remote and local workers have same host identity')
    if local['host_id'] != collector['host_id']:
        raise SystemExit('Topology mismatch: local worker and collector must share VM B')
    if remote['github_run_id'] != local['github_run_id'] or not remote['github_run_id']:
        raise SystemExit('Topology mismatch: workers do not belong to same GitHub run')
    if remote['github_job'] == local['github_job']:
        raise SystemExit('Topology mismatch: services report same GitHub job')
    if len({h['boot_id'] for h in health.values()}) != 3:
        raise SystemExit('Services do not have three distinct process boot IDs')
    results = []
    for name, expected, channel in original.SCENARIOS:
        report = original.run_case(name, expected, channel)
        results.append(report)
        print(f'{name:30s} PASS={report["pass"]} primary={report["primary_alarm"]} '
              f'full={report["full_alarm"]} receipts={report["receipt_rows"]}', flush=True)
    summary = {
        'schema_version': 'replaybench-pg-r1-experiment7-crosshost-v1',
        'deployment': 'two separate GitHub-hosted ubuntu-24.04 runner VMs',
        'worker0_remote_machine': remote,
        'worker1_local_machine': local,
        'collector_local_machine': collector,
        'remote_host_magicdns': remote_host,
        'transport': 'Tailscale encrypted peer network, HTTP application layer',
        'github_run_id': os.environ.get('GITHUB_RUN_ID', ''),
        'github_sha': os.environ.get('GITHUB_SHA', ''),
        'configurations': len(results),
        'passed': sum(r['pass'] for r in results),
        'positive_cases': sum(label for _, label, _ in original.SCENARIOS),
        'negative_cases': sum(1-label for _, label, _ in original.SCENARIOS),
        'host_python': platform.python_version(),
        'duration_s': round(time.monotonic() - started, 4),
        'scope_limitations': [
            'synthetic six-point workload; two GH virtual machines, not separately owned bare-metal hosts',
            'Tailscale overlay path may be direct or relayed; no route/latency isolation claim',
            'controlled receipt loss and delay, not real packet-loss or network-partition injection',
            'in-memory collector, no durable receipt ledger',
            'trusted deterministic reference; no independent downstream side-effect attestation',
            'not a production false-positive or sensitivity estimate',
            'non-idempotent retry intentionally duplicates execution',
        ],
    }
    (out / 'scenario_results.json').write_text(json.dumps(results, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    (out / 'experiment_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    if summary['passed'] != len(results):
        raise SystemExit(f'FAIL: {summary["passed"]}/{len(results)} case contracts passed')
    print(f'PASS: {len(results)}/{len(results)} cross-host scenarios on two GitHub VMs', flush=True)


if __name__ == '__main__':
    main()
