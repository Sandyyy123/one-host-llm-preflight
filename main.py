#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from cloudpilot.preflight import validate
from cloudpilot.benchmark import run


def main():
    parser=argparse.ArgumentParser(description='One-host planning and explicit endpoint measurement; no GPU provisioning')
    sub=parser.add_subparsers(dest='command',required=True)
    cfg=sub.add_parser('preflight');cfg.add_argument('config')
    bench=sub.add_parser('benchmark');bench.add_argument('--base-url',default='http://127.0.0.1:8000/v1');bench.add_argument('--model',required=True)
    bench.add_argument('--prompts',required=True);bench.add_argument('--concurrency',type=int,default=1);bench.add_argument('--max-tokens',type=int,default=32)
    bench.add_argument('--timeout',type=float,default=20);bench.add_argument('--allow-remote',action='store_true');bench.add_argument('--api-key-env')
    mode=bench.add_mutually_exclusive_group();mode.add_argument('--execute',action='store_true');mode.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    try:
        if args.command=='preflight':
            result=validate(json.loads(Path(args.config).read_text()))
            code=2 if result['status'] in ('invalid','estimate_exceeds_budget') else 0
        else:
            result=run(args.base_url,args.model,json.loads(Path(args.prompts).read_text()),args.concurrency,args.max_tokens,args.timeout,args.execute,args.allow_remote,args.api_key_env)
            code=2 if result.get('summary',{}).get('failed',0) else 0
        print(json.dumps(result,indent=2));return code
    except (ValueError,OSError,KeyError,TypeError) as exc:
        parser.error(str(exc))

if __name__=='__main__':
    raise SystemExit(main())
