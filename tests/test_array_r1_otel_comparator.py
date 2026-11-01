"""Independent audit invariants; tests never import the ReplayBench-PG validator."""
import unittest
from scripts.array_r1_otel_comparator import (
    POLICIES, SEEDS, WORKERS, NEGATIVE, POSITIVE, INCONCLUSIVE,
    make_case, independent_otel_audit, canonical_json, decode_exported,
    Receiver, otel_emit,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

class ComparatorContractTests(unittest.TestCase):
    def test_all_18_configurations_all_fault_contracts(self):
        tested=0
        for policy in POLICIES:
            for seed in SEEDS:
                for worker in WORKERS:
                    for scenario in NEGATIVE+POSITIVE+INCONCLUSIVE:
                        ref,rr,cfg,obs,rcts,cfg_obs=make_case(policy,seed,worker,scenario)
                        finding=independent_otel_audit(obs,rcts,cfg_obs,ref,rr,cfg)
                        expected=int(scenario in POSITIVE)
                        self.assertEqual(finding['alarm'],expected,(policy,seed,worker,scenario,finding))
                        tested+=1
        self.assertEqual(tested,342)

    def test_real_otlp_protobuf_export_and_normalized_delivery(self):
        receiver=Receiver.create()
        provider=TracerProvider(sampler=ALWAYS_ON)
        provider.add_span_processor(BatchSpanProcessor(
            OTLPSpanExporter(endpoint=f'http://127.0.0.1:{receiver.server.server_port}/v1/traces'),
            schedule_delay_millis=10000,max_export_batch_size=1024,max_queue_size=1024))
        tracer=provider.get_tracer('test')
        try:
            ref,rr,cfg,obs,rcts,cfg_obs=make_case('risk_proxy',1,1,'transport_dedup')
            spans,nbytes,_=otel_emit(tracer,provider,receiver,'test-dedup',obs,rcts,cfg_obs,
                                     duplicate_telemetry=True)
            pts,attempts,manifest=decode_exported(spans,'test-dedup')
            self.assertGreater(nbytes,100)
            self.assertEqual(pts,obs)
            self.assertEqual(attempts,rcts)
            self.assertEqual(manifest,cfg_obs)
            self.assertEqual(len([s for s in spans if s['name']=='rbpg.downstream_attempt']),len(rcts)+1)
            self.assertEqual(independent_otel_audit(pts,attempts,manifest,ref,rr,cfg)['alarm'],0)
        finally:
            provider.shutdown();receiver.close()

if __name__=='__main__':unittest.main()
