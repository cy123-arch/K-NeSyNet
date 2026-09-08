#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,math

def ss(x): return {v.strip() for v in str(x or '').split(';') if v.strip()}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('patient_outputs'); args=ap.parse_args()
    with open(args.patient_outputs,newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    f1s=[]; js=[]
    for r in rows:
        y=ss(r['true_labels']); p=ss(r['thresholded_positive_drugs']); tp=len(y&p); fp=len(p-y); fn=len(y-p)
        f1s.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0); js.append(tp/(tp+fp+fn) if tp+fp+fn else 0.0)
    mf=sum(f1s)/len(f1s); mj=sum(js)/len(js); bound=2*mj/(1+mj)
    print(f'n={len(rows)} mean_CR_F1={mf:.8f} mean_Jaccard={mj:.8f} Jensen_upper_bound_F1_given_mean_J={bound:.8f} gap={bound-mf:.8g}')
    if mf > bound + 1e-12: raise SystemExit('ERROR: F1/Jaccard aggregation is mathematically inconsistent')
if __name__=='__main__': main()
