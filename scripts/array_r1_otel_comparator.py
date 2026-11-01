#!/usr/bin/env python3
"""Experiment 6: fair OTLP/HTTP OpenTelemetry + independent policy-audit vs ReplayBench-PG.

This is a PREDECLARED synthetic controlled-fault benchmark, not an OTel feature ranking.
OTel instrumentation uses the Python SDK, ALWAYS_ON sampler and real OTLP/HTTP
protobuf exporter. A local HTTP receiver decodes actual exported protobuf bytes.
Both validators receive the same trusted reference. Fault labels are only for
POST-validation scoring; they are excluded from the validation call and OTel spans.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ALWAYS_ON

# Repository's real validator is imported ONLY for P, never for the OTel audit.
# --otel-only allows a local pipeline/SDK smoke test without the GitHub repo.

POLICIES=("risk_proxy","random","always")
SEEDS=(1,2,3)
WORKERS=(1,4)
N=12
NEGATIVE=("clean","reordered_delivery","timestamp_drift","transport_dedup")
POSITIVE=("lost_receipt","false_execution_log","unlogged_call","duplicate_attempt",
          "bad_correlation","unauthorized_execution","action_flip","row_drop",
          "row_duplicate","record_reassignment","config_corruption",
          "authorization_bit_corruption","execution_bit_corruption")
INCONCLUSIVE=("unobserved_external_effect","compromised_reference")
SCENARIOS=NEGATIVE+POSITIVE+INCONCLUSIVE
RECEIPT_FIELDS=("schema_version","run_id","configuration_id","execution_instance_id",
                "replay_point_id","correlation_id","downstream_operation","attempt_index",
                "execution_status")
POINT_FIELDS=("run_id","configuration_id","execution_instance_id","source_record_id",
              "replay_point_id","correlation_id","action","authorized_to_generate",
              "generation_invoked","unauthorized_invocation")


def canonical_json(obj):
    return json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False,default=str)


def sha256(obj):
    return hashlib.sha256(canonical_json(obj).encode('utf8')).hexdigest()


def sha256text(s):
    return hashlib.sha256(s.encode('utf8')).hexdigest()


def corr(eid,rid):
    return sha256text(f'{eid}|{rid}')[:32]


def rows_for(policy,seed,workers):
    c=f'cmp18-{policy}-s{seed}-w{workers}'
    eid=f'{c}-execution-1'
    rng=random.Random(seed)
    rows=[]
    for i in range(N):
        action=(int(i%3==0) if policy=='risk_proxy' else rng.randrange(2) if policy=='random' else 1)
        rid=f'{i}:synthetic-{i:04d}'
        rows.append({'run_id':c,'configuration_id':c,'execution_instance_id':eid,
                     'replay_point_id':rid,'source_record_id':f'synthetic-{i:04d}',
                     'correlation_id':corr(eid,rid),'action':action,
                     'authorized_to_generate':action,'generation_invoked':action,
                     'unauthorized_invocation':0})
    return rows


def receipts_for(rows):
    output=[]
    for row in rows:
        if not int(row['generation_invoked']):
            continue
        base={k:row[k] for k in ('run_id','configuration_id','execution_instance_id','replay_point_id','correlation_id')}
        base.update({'schema_version':'replaybench-pg-receipt-v2',
                     'downstream_operation':'deterministic_stub','attempt_index':1,'execution_status':'executed'})
        output.append({**base,'receipt_digest':sha256(base)})
    return output


def clone(value):
    return json.loads(json.dumps(value))


def make_case(policy,seed,workers,scenario):
    ref=rows_for(policy,seed,workers)
    rr=receipts_for(ref)
    manifest={'schema_version':'r1-otel-comparator-v1', 'policy_mode':policy,'seed':seed,
              'workers':workers,'workload':'12-synthetic-records', 'downstream':'deterministic_stub'}
    obs=clone(ref); ro=clone(rr); cfg=clone(manifest)
    a=next(i for i,r in enumerate(obs) if r['action']==1)
    z=next(i for i,r in enumerate(obs) if r['action']==0) if any(r['action']==0 for r in obs) else None
    rid=obs[a]['replay_point_id']; matching=next(j for j,r in enumerate(ro) if r['replay_point_id']==rid)
    if scenario=='lost_receipt': ro.pop(matching)
    elif scenario=='false_execution_log': ro.pop(matching)
    elif scenario=='unlogged_call': obs[a]['generation_invoked']=0
    elif scenario=='duplicate_attempt':
        extra=clone(ro[matching]);extra['attempt_index']=2;extra['receipt_digest']=sha256({k:extra[k] for k in RECEIPT_FIELDS});ro.append(extra)
    elif scenario=='bad_correlation':
        ro[matching]['correlation_id']='wrong-correlation';ro[matching]['receipt_digest']=sha256({k:ro[matching][k] for k in RECEIPT_FIELDS})
    elif scenario=='unauthorized_execution':
        if z is None: # always policy: turn permitted bit off without changing action.
            obs[a]['authorized_to_generate']=0
        else:
            obs[z]['generation_invoked']=1
            base={k:obs[z][k] for k in ('run_id','configuration_id','execution_instance_id','replay_point_id','correlation_id')}
            base.update({'schema_version':'replaybench-pg-receipt-v2','downstream_operation':'deterministic_stub','attempt_index':1,'execution_status':'executed'})
            ro.append({**base,'receipt_digest':sha256(base)})
    elif scenario=='action_flip':obs[a]['action']=0
    elif scenario=='row_drop':obs.pop(a)
    elif scenario=='row_duplicate':obs.insert(a+1,clone(obs[a]))
    elif scenario=='record_reassignment':
        obs[0]['source_record_id'],obs[1]['source_record_id']=obs[1]['source_record_id'],obs[0]['source_record_id']
    elif scenario=='config_corruption':cfg['workers']=workers+100
    elif scenario=='authorization_bit_corruption':obs[a]['authorized_to_generate']=0
    elif scenario=='execution_bit_corruption':obs[a]['generation_invoked']=0
    elif scenario=='compromised_reference':
        # Both the reference and observed world have been consistently altered;
        # observability and trusted-reference assumption fail. Deliberately coherent.
        i=next(i for i,r in enumerate(ref) if r['action']==0) if any(r['action']==0 for r in ref) else None
        if i is not None:
            ref[i]['source_record_id']='forged-'+ref[i]['source_record_id']
            obs[i]['source_record_id']=ref[i]['source_record_id']
        else:
            ref[0]['source_record_id']='forged-'+ref[0]['source_record_id']
            obs[0]['source_record_id']=ref[0]['source_record_id']
    # timestamp_drift, transport_dedup, unobserved_external_effect: no meaningful
    # mutation of validator-visible evidence after appropriate normalization.
    # Any truly unobserved external effect cannot be adjudicated without independent trust.
    return ref,rr,manifest,obs,ro,cfg


@dataclass
class Receiver:
    server: ThreadingHTTPServer
    thread: threading.Thread
    exports: list
    lock: threading.Lock

    @classmethod
    def create(cls):
        exports=[];lock=threading.Lock()
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path!='/v1/traces':
                    self.send_error(404);return
                size=int(self.headers['Content-Length'])
                raw=self.rfile.read(size)
                req=ExportTraceServiceRequest()
                try: req.ParseFromString(raw)
                except Exception:
                    self.send_error(400);return
                spans=[]
                for rs in req.resource_spans:
                    for ss in rs.scope_spans:
                        for s in ss.spans:
                            attrs={}
                            for kv in s.attributes:
                                value=kv.value
                                choice=value.WhichOneof('value')
                                attrs[kv.key]=getattr(value,choice) if choice in ('string_value','int_value','bool_value','double_value') else str(value)
                            spans.append({'name':s.name,'attributes':attrs,'status_code':int(s.status.code),
                                          'trace_id':s.trace_id.hex(),'span_id':s.span_id.hex(),
                                          'parent_span_id':s.parent_span_id.hex()})
                with lock: exports.append({'spans':spans,'wire_bytes':len(raw)})
                self.send_response(200)
                self.send_header('Content-Type','application/x-protobuf')
                self.end_headers()
                self.wfile.write(b'')
            def log_message(self,*args):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        return cls(server,thread,exports,lock)
    def collect(self,case_id):
        with self.lock:
            fragments=[e for e in self.exports if any(s['attributes'].get('rbpg.evidence_id')==case_id for s in e['spans'])]
        result=[s for e in fragments for s in e['spans'] if s['attributes'].get('rbpg.evidence_id')==case_id]
        return result,sum(e['wire_bytes'] for e in fragments)
    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=3)


def otel_emit(tracer,provider,receiver,case_id,obs,receipts,config,arrival_reverse=False,duplicate_telemetry=False,timestamp_shift=False):
    t0=time.perf_counter()
    cfg=canonical_json(config)
    with tracer.start_as_current_span('rbpg.case',attributes={'rbpg.evidence_id':case_id,'rbpg.manifest_json':cfg}):
        indexed=list(enumerate(obs))
        if arrival_reverse:indexed.reverse()
        for position,row in indexed:
            attr={'rbpg.evidence_id':case_id,'rbpg.position':position}
            attr.update({'rbpg.'+k:row[k] for k in POINT_FIELDS})
            if timestamp_shift:attr['rbpg.auxiliary_timestamp_offset_ms']=60000
            with tracer.start_as_current_span('rbpg.primary_point',attributes=attr):pass
        receipt_events=list(enumerate(receipts))
        if arrival_reverse:receipt_events.reverse()
        for ix,r in receipt_events:
            attr={'rbpg.evidence_id':case_id,'rbpg.receipt_index':ix}
            attr.update({'rbpg.'+k:r[k] for k in (*RECEIPT_FIELDS,'receipt_digest')})
            with tracer.start_as_current_span('rbpg.downstream_attempt',attributes=attr):pass
            if duplicate_telemetry and ix==0:
                # One duplicate telemetry report of the SAME attempt, not a second execution.
                with tracer.start_as_current_span('rbpg.downstream_attempt',attributes=attr):pass
    if not provider.force_flush(timeout_millis=30000):
        raise RuntimeError('OTel exporter force_flush failed')
    spans,nbytes=receiver.collect(case_id)
    if not spans:
        raise RuntimeError('OTLP exporter/collector did not deliver spans')
    return spans,nbytes,(time.perf_counter()-t0)*1000


def decode_exported(spans,case_id):
    points=[s for s in spans if s['name']=='rbpg.primary_point']
    attempts=[s for s in spans if s['name']=='rbpg.downstream_attempt']
    cases=[s for s in spans if s['name']=='rbpg.case']
    if len(cases)!=1:
        raise ValueError(f'Expected one OTel case root for {case_id}, got {len(cases)}')
    pts=[]
    for s in sorted(points,key=lambda s:int(s['attributes']['rbpg.position'])):
        d={k:s['attributes']['rbpg.'+k] for k in POINT_FIELDS}
        for k in ('action','authorized_to_generate','generation_invoked'):d[k]=int(d[k])
        pts.append(d)
    rcts=[];delivery_keys=set()
    for s in sorted(attempts,key=lambda s:int(s['attributes']['rbpg.receipt_index'])):
        d={k:s['attributes']['rbpg.'+k] for k in (*RECEIPT_FIELDS,'receipt_digest')}
        d['attempt_index']=int(d['attempt_index'])
        # Transport idempotency: same receipt with same attempt identity+digest is
        # telemetry redelivery, not a second downstream operation.
        delivery_key=(d['execution_instance_id'],d['replay_point_id'],d['attempt_index'],d['receipt_digest'])
        if delivery_key not in delivery_keys:
            delivery_keys.add(delivery_key);rcts.append(d)
    cfg=json.loads(cases[0]['attributes']['rbpg.manifest_json'])
    return pts,rcts,cfg


def otel_trace_only_alarm(spans):
    # Basic status/error-event operational tracing, NOT a policy integrity audit.
    return int(any(s['status_code']==2 for s in spans))


def independent_otel_audit(points,attempts,config,reference_points,reference_attempts,reference_config):
    """Independent policy-aware comparator; no reuse of ReplayBench validator functions.

    Equality tests are deliberately at least as strong as hashing semantic fields
    against the same reference; reference access is equal to P. Check channels
    are returned for diagnosis but not case fault label inference.
    """
    channels=[]; ids=set()
    def flag(channel,point=None):
        if channel not in channels:channels.append(channel)
        if point is not None: ids.add(str(point))
    if len(points)!=len(reference_points):flag('cardinality')
    if [int(p['action']) for p in points]!=[int(p['action']) for p in reference_points]:flag('ordered_action_sequence')
    ref_ids=[r['replay_point_id'] for r in reference_points]
    obs_ids=[r['replay_point_id'] for r in points]
    if obs_ids!=ref_ids:flag('record_order_or_identity')
    if len(obs_ids)!=len(set(obs_ids)):flag('duplicate_identity')
    rm={r['replay_point_id']:r for r in reference_points}
    for p in points:
        rid=p['replay_point_id']
        if rid not in rm:
            flag('unexpected_replay_point',rid);continue
        for k in POINT_FIELDS:
            if p[k]!=rm[rid][k]:flag('record_or_execution_mismatch',rid)
        if int(p['generation_invoked'])==1 and int(p['authorized_to_generate'])==0:
            flag('unauthorized_execution',rid)
    for rid in set(ref_ids)-set(obs_ids):flag('missing_replay_point',rid)
    if sha256(config)!=sha256(reference_config):flag('configuration_reference_mismatch')
    actual=defaultdict(list)
    for r in attempts:
        rid=str(r['replay_point_id']);actual[rid].append(r)
        if r['run_id']!=r['configuration_id'] or sha256({k:r[k] for k in RECEIPT_FIELDS})!=r['receipt_digest']:
            flag('receipt_digest_invalid',rid)
    expected=defaultdict(list)
    for r in reference_attempts:expected[str(r['replay_point_id'])].append(r)
    for rid in set(expected)|set(actual)|set(obs_ids):
        x=actual[rid];y=expected[rid]
        if len(x)!=len(y):flag('receipt_multiplicity_or_missing',rid)
        if sorted(canonical_json(r) for r in x)!=sorted(canonical_json(r) for r in y):
            flag('receipt_reference_mismatch',rid)
        op=next((p for p in points if p['replay_point_id']==rid),None)
        if op is None and x:flag('orphan_receipt',rid)
        if op is not None:
            if len(x)>1:flag('duplicate_actual_attempt',rid)
            if int(op['generation_invoked'])!=len(x):flag('primary_downstream_disagreement',rid)
            if int(op['authorized_to_generate'])==0 and x:flag('downstream_unauthorized',rid)
            if any(str(r['correlation_id'])!=str(op['correlation_id']) or str(r['execution_instance_id'])!=str(op['execution_instance_id']) for r in x):
                flag('correlation_or_instance_mismatch',rid)
    return {'alarm':int(bool(channels)),'channels':channels,'affected_replay_points':sorted(ids)}


_REFERENCE_CACHE={}

def p_validate(case_id,ref,ref_receipts,manifest,obs,attempts,config):
    from replaybench.generic_validator import prepare_reference_artifacts,validate_observed_artifacts
    reference_trace=pd.DataFrame(ref);reference_receipts=pd.DataFrame(ref_receipts)
    observed_trace=pd.DataFrame(obs);observed_receipts=pd.DataFrame(attempts)
    if observed_receipts.empty:
        observed_receipts=pd.DataFrame(columns=reference_receipts.columns)
    key=sha256({"reference_trace":ref,"reference_receipts":ref_receipts,"reference_manifest":manifest})
    prepared=_REFERENCE_CACHE.get(key)
    if prepared is None:
        prepared=prepare_reference_artifacts(reference_trace=reference_trace,
            reference_receipts=reference_receipts,reference_config_manifest=manifest)
        _REFERENCE_CACHE[key]=prepared
    return validate_observed_artifacts(evidence_id=case_id,observed_trace=observed_trace,
        observed_receipts=observed_receipts,observed_config_manifest=config,
        reference_trace=reference_trace,reference_receipts=reference_receipts,
        reference_config_manifest=manifest,prepared_reference=prepared).to_dict()


def measure_p(case_id,ref,ref_receipts,manifest,obs,attempts,config):
    t=time.perf_counter();d=p_validate(case_id,ref,ref_receipts,manifest,obs,attempts,config)
    return d,(time.perf_counter()-t)*1000


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+'\n',encoding='utf8')


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(output,otel_only=False,timing_reps=20):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    receiver=Receiver.create()
    provider=TracerProvider(resource=Resource.create({'service.name':'replaybench-r1-otel-comparator',
         'service.version':'1.0.0'}),sampler=ALWAYS_ON)
    exporter=OTLPSpanExporter(endpoint=f'http://127.0.0.1:{receiver.server.server_port}/v1/traces',timeout=10)
    provider.add_span_processor(BatchSpanProcessor(exporter,schedule_delay_millis=10000,
           max_queue_size=65536,max_export_batch_size=1024))
    tracer=provider.get_tracer('array_r1_otel_comparator', '1.0.0')
    results=[]; score_manifest=[]; timing=[]
    raw_file=(out/"observed_evidence.jsonl").open("w",encoding="utf8")
    export_file=(out/"exported_otel_spans.jsonl").open("w",encoding="utf8")
    try:
        k=0
        for policy in POLICIES:
            for seed in SEEDS:
                for workers in WORKERS:
                    for scenario in SCENARIOS:
                        k+=1;case_id=f'unit{k:04d}'
                        ref,rr,manifest,obs,rcts,config=make_case(policy,seed,workers,scenario)
                        spans,nbytes,emit_ms=otel_emit(tracer,provider,receiver,case_id,obs,rcts,config,
                            arrival_reverse=(scenario=='reordered_delivery'),
                            duplicate_telemetry=(scenario=='transport_dedup'),
                            timestamp_shift=(scenario=='timestamp_drift'))
                        t0=time.perf_counter()
                        decoded,attempts,decoded_cfg=decode_exported(spans,case_id)
                        b1=independent_otel_audit(decoded,attempts,decoded_cfg,ref,rr,manifest)
                        b1_ms=(time.perf_counter()-t0)*1000
                        b0=otel_trace_only_alarm(spans)
                        if decoded!=obs or attempts!=rcts or decoded_cfg!=config:
                            raise AssertionError(f'OTLP exporter evidence fidelity mismatch {case_id}')
                        p=None;p_ms=None
                        if not otel_only:
                            p,p_ms=measure_p(case_id,ref,rr,manifest,obs,rcts,config)
                        expected=int(scenario in POSITIVE)
                        normal=scenario not in INCONCLUSIVE
                        if normal and b1['alarm']!=expected:
                            raise AssertionError(f'B1 wrong {case_id}: {scenario}, got {b1}')
                        if p is not None and normal and int(p['full_validator_triggered'])!=expected:
                            raise AssertionError(f'P wrong {case_id}: {scenario}, got full={p["full_validator_triggered"]}')
                        if p is not None and normal and int(p['full_validator_triggered'])!=int(b1['alarm']):
                            raise AssertionError(f'B1/P mismatch {case_id}: {scenario}')
                        results.append({'case_id':case_id,'policy':policy,'seed':seed,'workers':workers,
                            'class':('positive' if expected else 'negative' if normal else 'observational_limit'),
                            'b0_alarm':b0,'b1_alarm':b1['alarm'],
                            'p_full_alarm':None if p is None else int(p['full_validator_triggered']),
                            'p_primary_alarm':None if p is None else int(p['primary_validator_triggered']),
                            'b1_channels':';'.join(b1['channels']),'otel_spans':len(spans),
                            'otel_wire_bytes':nbytes,'otel_export_ms':round(emit_ms,4),
                            'b1_audit_ms':round(b1_ms,4),
                            'p_adapter_validation_ms':None if p_ms is None else round(p_ms,4),
                            'b1_ids_count':len(b1['affected_replay_points'])})
                        raw_file.write(canonical_json({"case_id":case_id,"observed_trace":obs,
                            "observed_receipts":rcts,"observed_config":config,"reference_trace":ref,
                            "reference_receipts":rr,"reference_config":manifest})+"\n")
                        export_file.write(canonical_json({"case_id":case_id,"exported_spans":spans})+"\n")
                        score_manifest.append({'case_id':case_id,'injected_case':scenario,
                                               'expected_observable_alarm':expected if normal else None})
        # Measured paired validation overhead after one warmup, using fixed 6 configurations
        # and two predefined cases. OTel emission/export and local audits are separate stages.
        if timing_reps:
            for policy in POLICIES:
                for workers in WORKERS:
                    seed=1
                    for scenario in ('clean','lost_receipt'):
                        ref,rr,man,obs,rcts,cfg=make_case(policy,seed,workers,scenario)
                        for rep in range(-1,timing_reps):
                            cid=f'timing-{policy}-{workers}-{scenario}-{rep}'
                            spans,nbytes,emit_ms=otel_emit(tracer,provider,receiver,cid,obs,rcts,cfg)
                            t=time.perf_counter()
                            pts,at,cf=decode_exported(spans,cid)
                            b1=independent_otel_audit(pts,at,cf,ref,rr,man)
                            b1_ms=(time.perf_counter()-t)*1000
                            p_ms=None
                            if not otel_only:
                                p,p_ms=measure_p(cid,ref,rr,man,obs,rcts,cfg)
                                if int(p['full_validator_triggered'])!=b1['alarm']:
                                    raise AssertionError('Timed B1/P disagreement')
                            if rep>=0:
                                timing.append({'policy':policy,'workers':workers,'scenario':scenario,'rep':rep,
                                    'otel_emit_export_ms':round(emit_ms,4),'otel_b1_decode_audit_ms':round(b1_ms,4),
                                    'p_adapter_validation_ms':None if p_ms is None else round(p_ms,4),
                                    'otlp_wire_bytes':nbytes,'process_rss_bytes':psutil.Process().memory_info().rss})
        if not provider.force_flush(timeout_millis=30000): raise RuntimeError('final exporter flush failed')
    finally:
        raw_file.close();export_file.close()
        provider.shutdown();receiver.close()
    import importlib.metadata as md
    versions={package:md.version(package) for package in ('opentelemetry-sdk','opentelemetry-api',
                'opentelemetry-exporter-otlp-proto-http','opentelemetry-proto','pandas','numpy','psutil')}
    grouped={}
    for category in ('positive','negative','observational_limit'):
        subset=[r for r in results if r['class']==category]
        grouped[category]={'n':len(subset), 'b0_alarms':sum(x['b0_alarm'] for x in subset),
                           'b1_alarms':sum(x['b1_alarm'] for x in subset),
                           'p_alarms':None if otel_only else sum(x['p_full_alarm'] for x in subset)}
    report={'schema_version':'r1-otel-comparison-v1','status':'otlp_sdk_export_implemented',
        'case_count':len(results),'base_configurations':len(POLICIES)*len(SEEDS)*len(WORKERS),
        'scenarios_per_configuration':len(SCENARIOS),'scenarios':list(SCENARIOS),
        'groups':grouped,'otel_only':otel_only,'timing_repetitions_per_selected_pair':timing_reps,
        'timing_measured_rows':len(timing),'environment':{'python':platform.python_version(),
            'os':platform.platform(),'versions':versions,'git_sha':os.environ.get('GITHUB_SHA',''),
            'otel_sampling':'ALWAYS_ON','otlp_transport':'HTTP/protobuf','exporter':'OTLPSpanExporter',
            'collector':'local HTTP OTLP protobuf receiver','processor':'BatchSpanProcessor'},
        'limits':['Synthetic 12-record configurations, not MELD or MetroPT-3',
         'OTel policy-aware auditor is separately implemented bespoke logic on OTel telemetry, not a native built-in feature',
         'Both auditing systems use same trusted reference',
         'All evaluations on one host, no independent trust boundary',
         'Inconclusive cases deliberately have no observable difference under declared assumptions; no external effect is independently observed',
         'OTel B0 basic span-status baseline lacks policy-specific application audit',
         'Timing separates OTel emission/export, OTel decoder+auditor, and P input adaptation+validation; these stages are not matched end-to-end costs',
         'Data from repeated scenarios sharing 18 configurations are statistically dependent',
         'Lost receipt and false execution log have the same observable signature in this controlled producer',
         'Duplicate telemetry reports are deduplicated by same attempt index and digest, but coordinated exact duplicate actual operations are not distinguishable']}
    with (out/'per_case.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=results[0].keys());w.writeheader();w.writerows(results)
    with (out/'timing_raw.csv').open('w',newline='',encoding='utf8') as f:
        if timing:
            w=csv.DictWriter(f,fieldnames=timing[0].keys());w.writeheader();w.writerows(timing)
    write_json(out/'experiment_summary.json',report)
    write_json(out/'ground_truth_post_scoring.json',score_manifest)
    manifest={'sha256':{p.name:sha_file(p) for p in sorted(out.iterdir()) if p.is_file()}}
    write_json(out/'file_hashes.json',manifest)
    print(json.dumps({'case_count':len(results),'groups':grouped,'timing_rows':len(timing),
                     'otel_only':otel_only},sort_keys=True,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='array_r1_otel_results')
    p.add_argument('--otel-only',action='store_true',help='OTel pipeline smoke; no P result claimed')
    p.add_argument('--timing-reps',type=int,default=20)
    a=p.parse_args()
    if a.timing_reps<0:raise SystemExit('timing-reps must be nonnegative')
    run(a.output,a.otel_only,a.timing_reps)
