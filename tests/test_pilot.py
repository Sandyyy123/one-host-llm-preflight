import copy
import json
import os
import threading
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from cloudpilot.preflight import validate
from cloudpilot.benchmark import run, summarize, endpoint_url

ROOT=Path(__file__).resolve().parents[1]

@contextmanager
def fixture():
    class Handler(BaseHTTPRequestHandler):
        calls=[]
        def log_message(self,*args):
            pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            type(self).calls.append({'path':self.path,'body':body})
            mode=body['model']
            if mode=='redirect':
                self.send_response(302);self.send_header('Location','http://127.0.0.1:1/never');self.end_headers();return
            if mode=='error':
                self.send_response(500);self.end_headers();self.wfile.write(b'synthetic secret error');return
            response={'choices':[{'message':{'content':'synthetic reply'}}]}
            if mode!='missing_usage':response['usage']={'completion_tokens':7}
            if mode=='bad_usage':response['usage']={'completion_tokens':-1}
            raw=b'not-json' if mode=='malformed' else json.dumps(response).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:yield f'http://127.0.0.1:{server.server_port}/v1',Handler
    finally:server.shutdown();server.server_close();thread.join()

class PreflightTests(unittest.TestCase):
    def setUp(self):self.config=json.loads((ROOT/'examples/synthetic_config.json').read_text())
    def test_budget_is_estimate_not_approval(self):
        r=validate(self.config);self.assertEqual(r['status'],'needs_runtime_verification');self.assertAlmostEqual(r['estimated_margin_per_gpu_gib'],4.6);self.assertFalse(r['deployment_verified'])
    def test_insufficient_memory_is_rejected(self):
        self.config['weights_total_gib']=60;self.assertEqual(validate(self.config)['status'],'estimate_exceeds_budget')
    def test_parallelism_cannot_exceed_host(self):
        self.config['tensor_parallel']=4;self.assertEqual(validate(self.config)['status'],'invalid')
    def test_mutable_revision_rejected(self):
        self.config['model_revision']='main';self.assertEqual(validate(self.config)['status'],'invalid')
    def test_nan_bool_negative_and_fraction_rejected(self):
        for key,value in [('gpu_memory_gib',float('nan')),('gpu_count',True),('kv_cache_per_gpu_gib',-1),('gpu_memory_fraction',1.1)]:
            cfg=copy.deepcopy(self.config);cfg[key]=value
            with self.subTest(key=key):self.assertEqual(validate(cfg)['status'],'invalid')

class ClientTests(unittest.TestCase):
    def test_dry_run_makes_no_request_or_secret_read(self):
        with patch('urllib.request.build_opener',side_effect=AssertionError('network')):
            r=run('http://127.0.0.1:1/v1','demo',['x'],api_key_env='ABSENT_SECRET')
        self.assertEqual(r['mode'],'dry_run')
    def test_remote_needs_explicit_permission_and_tls(self):
        with self.assertRaises(ValueError):endpoint_url('https://example.com/v1')
        with self.assertRaises(ValueError):endpoint_url('http://example.com/v1',True)
        self.assertEqual(endpoint_url('https://example.com/v1',True),'https://example.com/v1/chat/completions')
    def test_credentials_query_fragment_rejected(self):
        for url in ['http://user:secret@localhost/v1','http://localhost/v1?token=x','http://localhost/v1#x']:
            with self.subTest(url=url),self.assertRaises(ValueError):endpoint_url(url)
    def test_success_usage_and_actual_api_payload(self):
        with fixture() as (url,server):
            r=run(url,'demo',['private prompt','second'],concurrency=2,execute=True)
            self.assertEqual(len(server.calls),2);self.assertEqual(server.calls[0]['path'],'/v1/chat/completions')
            self.assertFalse(server.calls[0]['body']['stream'])
        self.assertEqual(r['summary']['successful'],2);self.assertEqual(r['summary']['known_success_output_tokens'],14)
        self.assertNotIn('private prompt',json.dumps(r));self.assertNotIn('synthetic reply',json.dumps(r));self.assertIsNone(r['summary']['time_to_first_token'])
    def test_missing_usage_never_becomes_zero_throughput(self):
        with fixture() as (url,_):r=run(url,'missing_usage',['x'],execute=True)
        self.assertEqual(r['summary']['successful'],1);self.assertFalse(r['summary']['success_usage_complete']);self.assertIsNone(r['summary']['successful_output_tokens_per_wall_second'])
    def test_failures_reported_without_body(self):
        with fixture() as (url,_):
            for mode in ['error','malformed','bad_usage','redirect']:
                with self.subTest(mode=mode):
                    r=run(url,mode,['x'],execute=True);self.assertEqual(r['summary']['failed'],1);self.assertNotIn('synthetic secret',json.dumps(r));self.assertIsNone(r['summary']['p95_success_latency_s'])
    def test_percentile_and_walltime_denominator(self):
        rows=[{'ok':True,'latency_s':1,'output_tokens':10},{'ok':True,'latency_s':3,'output_tokens':20},{'ok':False,'latency_s':5,'output_tokens':None}]
        r=summarize(rows,10);self.assertEqual(r['successful_output_tokens_per_wall_second'],3);self.assertEqual(r['p95_success_latency_s'],3);self.assertEqual(r['failed'],1)
    def test_bounded_workload(self):
        for kw in [{'concurrency':9},{'max_tokens':1025},{'timeout':float('inf')}]:
            with self.subTest(kw=kw),self.assertRaises(ValueError):run('http://localhost:1/v1','demo',['x'],**kw)
    def test_missing_requested_key_fails_before_requests(self):
        with patch.dict(os.environ,{},clear=True),self.assertRaises(ValueError):
            run('http://localhost:1/v1','demo',['x'],execute=True,api_key_env='MISSING_KEY')

if __name__=='__main__':unittest.main()
