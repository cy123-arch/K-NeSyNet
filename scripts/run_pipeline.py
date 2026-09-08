from __future__ import annotations
import argparse, subprocess, sys

def run(cmd):
    print('+',' '.join(cmd)); subprocess.run(cmd,check=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/final.yaml'); ap.add_argument('--smoke-test',action='store_true'); args=ap.parse_args()
    if args.smoke_test:
        run([sys.executable,'scripts/smoke_pipeline.py'])
        return
    print('Full-data workflow is manifest driven. Execute the numbered commands in PIPELINE_COMMANDS.md after placing authorized source files at the configured locations.')
if __name__=='__main__': main()
