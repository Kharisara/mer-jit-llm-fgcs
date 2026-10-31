#!/usr/bin/env python3
"""Controlled multi-container transport validation using the actual ReplayBench-PG validator.

Synthetic six-point workload. Scoring labels are withheld from validation calls.
Runs against HTTP processes across Docker network on the SAME HOST.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, sys, time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from replaybench.generic_validator import prepare_reference_artifacts, validate_observed_artifacts
from replaybench.integrity import make_correlation_id, sha256_json

WORKERS=('http://127.0.0.1:18761','http://127.0.0.1:18762')
COLLECTOR='http://127.0.0.1:18766'
SCENARIOS=(
    ('clean',0,'clean'),
    ('reordered',0,'clean'),
    ('delayed',0,'clean'),
    ('transport_redelivery',0,'clean'),
    ('lost_receipt',1,'missing_receipts'),
    ('changed_correlation',1,'mismatched_correlation_ids'),
    ('non_idempotent_retry',1,'duplicate_downstream_calls'),
    ('worker_crash_after_receipt',1,'unlogged_downstream_calls'),
)

def request_json(base, path, data=None, timeout=4):
    body=None if data is None else json.dumps(data,separators=(',',':')).encode('utf-8')
    req=Request(base+path,data=body,headers={'Content-Type':'application/json'} if body else {},method='POST' if body else 'GET')
    try:
        with urlopen(req,timeout=timeout) as resp:
            return resp.status,json.loads(resp.read())
    except HTTPError as exc:
        return exc.code,json.loads(exc.read())

def wait_healthy(base, until=15):
    deadline=time.monotonic()+until
    while time.monotonic()<deadline:
        try:
            status, response=request_json(base,'/health',timeout=2)
            if status==200 and response.get('ok'): return response
        except (OSError,ValueError): pass
        time.sleep(.1)
    raise RuntimeError(f'Service unavailable: {base}')

def wait_receipts(required,until=12):
    deadline=time.monotonic()+until
    while time.monotonic()<deadline:
        _, response=request_json(COLLECTOR,'/receipts')
        if response['count']>=required:
            time.sleep(.1)
            _, response=request_json(COLLECTOR,'/receipts')
            return response['receipts']
        time.sleep(.05)
    raise RuntimeError(f'Only {response["count"]} receipts, required {required}')

def reset_all():
    for base in (*WORKERS,COLLECTOR):
        wait_healthy(base)
        status, result=request_json(base,'/reset',{})
        assert status==200,(base,status,result)

def build_trace(instance):
    rows=[]
    for i in range(6):
        replay_id=f'{i}:synthetic-{i:04d}'
        action=int(i<4)
        config_id='rbpg-dist-six-point-v1'
        rows.append({'run_id':config_id,'configuration_id':config_id,
            'execution_instance_id':instance,'source_record_id':f'synthetic-{i:04d}',
            'replay_point_id':replay_id,
            'correlation_id':make_correlation_id(instance,replay_id),
            'action':action,'authorized_to_generate':action,
            'generation_invoked':action,'unauthorized_invocation':0})
    return pd.DataFrame(rows)

def expected_receipts(trace):
    receipts=[]
    for row in trace.to_dict(orient='records'):
        if not row['action']:continue
        semantic={
            'schema_version':'replaybench-pg-receipt-v2',
            'run_id':row['run_id'],'configuration_id':row['configuration_id'],
            'execution_instance_id':row['execution_instance_id'],
            'replay_point_id':row['replay_point_id'],
            'correlation_id':row['correlation_id'],
            'downstream_operation':'deterministic_stub',
            'attempt_index':1,'execution_status':'executed',
        }
        receipts.append({**semantic,'receipt_digest':sha256_json(semantic)})
    return pd.DataFrame(receipts)

def dispatch(row,worker_index,mode='normal',delay_ms=0):
    req={k:str(row[k]) for k in ('run_id','configuration_id','execution_instance_id','replay_point_id','correlation_id')}
    req.update({'downstream_operation':'deterministic_stub','mode':mode,'delay_ms':delay_ms})
    return request_json(WORKERS[worker_index],'/execute',req,timeout=5)

def validate(evidence_id,trace,receipt_rows,reference_trace,reference_receipts,manifest,prepared):
    observed=pd.DataFrame(receipt_rows)
    if observed.empty:observed=pd.DataFrame(columns=reference_receipts.columns)
    return validate_observed_artifacts(
        evidence_id=evidence_id,
        observed_trace=trace,
        observed_receipts=observed,
        observed_config_manifest=manifest,
        reference_trace=reference_trace,
        reference_receipts=reference_receipts,
        reference_config_manifest=manifest,
        prepared_reference=prepared,
    ).to_dict()

def run_case(name,should_detect,expected_channel):
    reset_all()
    instance='r1-dist-'+name
    trace=build_trace(instance)
    reference_trace=trace.copy(deep=True)
    ref_receipts=expected_receipts(reference_trace)
    manifest={'schema_version':'r1-multi-container-v1', 'workload':'six-synthetic-points',
              'workers':2,'downstream':'deterministic_stub', 'transport':'http-docker-bridge'}
    prepared=prepare_reference_artifacts(reference_trace=reference_trace,
        reference_receipts=ref_receipts,reference_config_manifest=manifest)
    events=[]
    original_boot=None
    for i in range(4):
        row=trace.iloc[i]
        worker=i%2
        mode='normal'; delay=0
        if name=='reordered' and i==0: mode='reordered'; delay=500
        if name=='reordered' and i==2: mode='reordered'; delay=200
        if name=='delayed' and i==0: mode='delay_receipt'; delay=1000
        if name=='delayed' and i==2: mode='delay_receipt'; delay=700
        if name=='transport_redelivery' and i==0: mode='redeliver'; delay=100
        if name=='lost_receipt' and i==0: mode='drop_receipt'
        if name=='changed_correlation' and i==0: mode='bad_correlation'
        if name=='non_idempotent_retry' and i==0: mode='ack_lost'
        if name=='worker_crash_after_receipt' and i==0:
            mode='crash_after_receipt'; original_boot=wait_healthy(WORKERS[worker])['boot_id']
        try:
            code,response=dispatch(row,worker,mode,delay)
        except (URLError, OSError,TimeoutError,ValueError) as exc:
            code=0;response={'transport_error':type(exc).__name__}
        events.append({'row':i,'worker':worker,'mode':mode,'status':code,'outcome':response})
        if name=='non_idempotent_retry' and i==0:
            if code!=503:raise AssertionError('Expected lost acknowledgement simulation')
            code2,response2=dispatch(row,worker,'normal')
            events.append({'row':i,'worker':worker,'mode':'retry_after_lost_ack','status':code2})
            if code2!=200:raise AssertionError('Retry failed')
        elif name=='worker_crash_after_receipt' and i==0:
            # Primary cannot affirm success without acknowledgement. Execution did occur;
            # receipt survived worker death in independent collector process.
            trace.at[i,'generation_invoked']=0
            boot=wait_healthy(WORKERS[worker],until=15)['boot_id']
            if boot==original_boot:
                raise AssertionError('Worker process did not restart after crash')
        elif code!=200:
            raise AssertionError(f'Unexpected HTTP response {code} for {name} row {i}: {response}')
    want=3 if name=='lost_receipt' else (5 if name=='non_idempotent_retry' else 4)
    raw=wait_receipts(want)
    if len(raw)!=want:raise AssertionError(f'{name}: expected exactly {want} independent receipts, got {len(raw)}')
    findings=validate(name,trace,raw,reference_trace,ref_receipts,manifest,prepared)
    primary=int(findings['primary_validator_triggered'])
    full=int(findings['full_validator_triggered'])
    channel=int(findings.get(expected_channel,0)) if expected_channel!='clean' else None
    ok=(primary==0 and full==should_detect and (channel is None or channel>=1))
    report={'scenario':name, 'expected_full_alarm':should_detect,
            'expected_channel':expected_channel,'pass':ok,'primary_alarm':primary,
            'full_alarm':full,'receipt_rows':len(raw),
            'receipt_digest_rows_valid':findings['receipt_digest_rows_valid'],
            'receipt_validation_passed':findings['receipt_validation_passed'],
            'missing_receipts':findings['missing_receipts'],
            'unlogged_downstream_calls':findings['unlogged_downstream_calls'],
            'duplicate_downstream_calls':findings['duplicate_downstream_calls'],
            'mismatched_correlation_ids':findings['mismatched_correlation_ids'],
            'record_hash_match':findings['record_trace_hash_match'],
            'config_hash_match':findings['config_manifest_hash_match'],
            'events':events}
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    base_health={base:wait_healthy(base) for base in (*WORKERS,COLLECTOR)}
    if len({value['boot_id'] for value in base_health.values()}) != 3:
        raise SystemExit('Services did not report distinct process boot identifiers')
    results=[]
    started=time.monotonic()
    for name, expected, channel in SCENARIOS:
        r=run_case(name,expected,channel)
        results.append(r)
        print(f'{name:30s} PASS={r["pass"]} primary={r["primary_alarm"]} full={r["full_alarm"]} receipts={r["receipt_rows"]}',flush=True)
    config={'schema_version':'replaybench-pg-r1-multi-container-experiment-v1',
            'deployment':'three-service Docker bridge; one physical GitHub-hosted machine',
            'services':base_health,'source_commit':'from GITHUB_SHA env or git',
            'run_count':len(results),'passed':sum(x['pass'] for x in results),
            'positive_cases':sum(e for _,e,_ in SCENARIOS),
            'negative_cases':sum(1-e for _,e,_ in SCENARIOS),
            'host_python':platform.python_version(),
            'duration_s':round(time.monotonic()-started,4),
            'scope_limitations':['synthetic six-point workload','same-host containers, not independent machines',
                'HTTP, no cryptographic transport attestation','controlled transport and lifecycle faults',
                'trusted expected reference derived from deterministic execution contract',
                'no adversarial coordinated falsification or verified external side effects']}
    import os
    config['github_sha']=os.environ.get('GITHUB_SHA','')
    (out/'scenario_results.json').write_text(json.dumps(results,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    (out/'experiment_summary.json').write_text(json.dumps(config,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    if config['passed']!=len(results):
        raise SystemExit(f'FAIL: {config["passed"]}/{len(results)} passed')
    print('PASS: all',len(results),'scenario contracts; NO multi-machine claim',flush=True)

if __name__=='__main__':main()
