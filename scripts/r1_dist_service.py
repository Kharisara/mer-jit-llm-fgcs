#!/usr/bin/env python3
"""Controlled transport boundary services for ReplayBench-PG Reviewer 1.

This is a deliberately bounded experiment: separate containerized HTTP processes
on a single GitHub-hosted machine; no TLS, Byzantine faults, or persisted effects.
"""
from __future__ import annotations
import argparse, hashlib, json, os, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

RECEIPT_SCHEMA = 'replaybench-pg-receipt-v2'
SERVICE_BOOT_ID = str(uuid.uuid4())

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
             separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()

class Store:
    def __init__(self):
        self.lock = threading.RLock()
        self.receipts = []
        self.deliveries = set()
        self.attempts = {}
        self.idx = 0
    def reset(self):
        with self.lock:
            self.receipts.clear(); self.deliveries.clear(); self.attempts.clear(); self.idx = 0

store = Store()

def post_json(url, payload, timeout=4):
    data = json.dumps(payload, separators=(',', ':')).encode('utf-8')
    req = Request(url, data=data, headers={'Content-Type':'application/json'}, method='POST')
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())

def deliver(payload, delay_ms, duplicates=1):
    if delay_ms:
        time.sleep(float(delay_ms)/1000)
    for _ in range(duplicates):
        try:
            post_json(os.environ.get('RECEIPT_ENDPOINT','http://127.0.0.1:18766/receipt'), payload)
        except Exception as exc:
            # Transport failure is intentionally not disguised as successful delivery.
            print('RECEIPT_SEND_FAILED', type(exc).__name__, str(exc), flush=True)

def build_receipt(payload):
    keys = ('run_id','configuration_id','execution_instance_id',
            'replay_point_id','correlation_id','downstream_operation')
    context = {k:str(payload[k]) for k in keys}
    if context['run_id'] != context['configuration_id']:
        raise ValueError('run_id alias mismatch')
    attempt_key = (context['execution_instance_id'], context['replay_point_id'])
    with store.lock:
        attempt = store.attempts.get(attempt_key, 0) + 1
        store.attempts[attempt_key] = attempt
        store.idx += 1
        index = store.idx
    sem = {'schema_version': RECEIPT_SCHEMA, **context,
           'attempt_index':attempt, 'execution_status':'executed'}
    if payload.get('mode') == 'bad_correlation':
        sem['correlation_id'] = 'mismatched-'+sem['correlation_id']
    receipt = {**sem, 'receipt_event_index':index,
               'execution_timestamp_ns':time.time_ns(),
               'execution_monotonic_ns':time.monotonic_ns(),
               'receipt_digest':canonical_hash(sem)}
    return receipt

class API(BaseHTTPRequestHandler):
    def log_message(self, *_):
        return
    def send_json(self, code, data):
        b=json.dumps(data, separators=(',',':')).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        if self.path == '/health':
            return self.send_json(200, {'ok':True,'role':self.server.role,'service_pid':os.getpid(),'boot_id':SERVICE_BOOT_ID})
        if self.path == '/receipts' and self.server.role == 'collector':
            with store.lock: rows=[dict(x) for x in store.receipts]
            return self.send_json(200, {'receipts':rows,'count':len(rows)})
        return self.send_json(404, {'error':'not_found'})
    def do_POST(self):
        try:
            n=int(self.headers.get('Content-Length','0'))
            if n<0 or n>200000: raise ValueError('oversized request')
            data=json.loads(self.rfile.read(n)) if n else {}
            if self.path=='/reset':
                store.reset(); return self.send_json(200, {'ok':True})
            if self.path=='/receipt' and self.server.role=='collector':
                delivery_id=str(data['delivery_id'])
                receipt=data['receipt']
                with store.lock:
                    duplicate=delivery_id in store.deliveries
                    if not duplicate:
                        store.deliveries.add(delivery_id)
                        store.receipts.append(dict(receipt))
                return self.send_json(200, {'ok':True,'transport_duplicate':duplicate})
            if self.path=='/execute' and self.server.role=='worker':
                receipt=build_receipt(data)
                mode=data.get('mode','normal')
                if mode=='drop_receipt':
                    return self.send_json(200, {'executed':True,'receipt_transport':'dropped'})
                envelope={'delivery_id':str(uuid.uuid4()),'receipt':receipt}
                if mode in ('delay_receipt','reordered','redeliver'):
                    delay=int(data.get('delay_ms',150))
                    copies=2 if mode=='redeliver' else 1
                    threading.Thread(target=deliver,args=(envelope,delay,copies),daemon=True).start()
                else:
                    deliver(envelope,0)
                if mode=='ack_lost':
                    return self.send_json(503, {'executed':True,'acknowledgement':'lost'})
                if mode=='crash_after_receipt':
                    os._exit(47)
                return self.send_json(200, {'executed':True})
            return self.send_json(404, {'error':'not_found'})
        except (ValueError, KeyError, TypeError) as exc:
            return self.send_json(400, {'error':str(exc)})

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--role', choices=['collector','worker'],required=True)
    parser.add_argument('--port',type=int,required=True)
    args=parser.parse_args()
    server=ThreadingHTTPServer(('0.0.0.0', args.port),API)
    server.daemon_threads=True
    server.role=args.role
    print('LISTEN',args.role,args.port,flush=True)
    server.serve_forever(poll_interval=0.1)
