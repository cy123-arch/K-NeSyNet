#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,numpy as np
from sklearn.metrics import f1_score

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--npz',required=True,help='NPZ with y_true [N,80], scores [N,80]'); ap.add_argument('--out',required=True); args=ap.parse_args()
    z=np.load(args.npz); y=z['y_true']; s=z['scores']; grid=np.round(np.arange(0.05,0.951,0.01),2)
    vals=[f1_score(y,(s>=t).astype(int),average='samples',zero_division=0) for t in grid]; best=int(np.argmax(vals))
    with open(args.out,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f); w.writerow(['threshold','validation_samples_f1','selected']);
        for i,(t,v) in enumerate(zip(grid,vals)): w.writerow([f'{t:.2f}',f'{v:.10f}',int(i==best)])
    print({'selected_threshold':float(grid[best]),'objective':'validation sample-averaged multilabel F1 over all 80 candidate drugs','grid':'0.05..0.95 step 0.01'})
if __name__=='__main__': main()
