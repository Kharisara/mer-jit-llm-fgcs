#!/usr/bin/env python3
"""Authenticated cross-host adapter for the unchanged R1 HTTP service.

Only synthetic experiment traffic is accepted. Shared-secret application checks
supplement (not replace) the Tailscale encrypted connection and tailnet ACLs.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import platform
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

from scripts import r1_dist_service as original

SECRET = os.environ.get('R1_CROSSHOST_TOKEN', '')
if len(SECRET) < 32:
    raise SystemExit('R1_CROSSHOST_TOKEN must be a random secret of at least 32 characters')


def post_json(url: str, payload: dict, timeout: float = 4):
    encoded = json.dumps(payload, separators=(',', ':')).encode('utf-8')
    request = Request(url, data=encoded, method='POST', headers={
        'Content-Type': 'application/json', 'X-R1-Crosshost-Token': SECRET,
    })
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


# The original service's delivery handler uses this module-level function.
original.post_json = post_json


class AuthenticatedAPI(original.API):
    def authenticate(self) -> bool:
        supplied = self.headers.get('X-R1-Crosshost-Token', '')
        if hmac.compare_digest(SECRET, supplied):
            return True
        self.send_json(403, {'error': 'forbidden'})
        return False

    def do_GET(self):
        if not self.authenticate():
            return
        if self.path == '/health':
            return self.send_json(200, {
                'ok': True,
                'role': self.server.role,
                'service_pid': os.getpid(),
                'boot_id': original.SERVICE_BOOT_ID,
                'host_id': platform.node(),
                'github_job': os.getenv('GITHUB_JOB', ''),
                'github_run_id': os.getenv('GITHUB_RUN_ID', ''),
            })
        return super().do_GET()

    def do_POST(self):
        if not self.authenticate():
            return
        if self.path == '/complete' and self.server.role == 'worker':
            self.send_json(200, {'ok': True, 'shutdown': 'requested'})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        return super().do_POST()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=['worker', 'collector'], required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('0.0.0.0', args.port), AuthenticatedAPI)
    server.daemon_threads = True
    server.role = args.role
    print('LISTEN', args.role, args.port, platform.node(), flush=True)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
