from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def _load_feature_npz(path, key='features'):
    z=np.load(path,allow_pickle=True)
    ids=z['case_id'].astype(str)
    feats=z[key]
    return {c:feats[i] for i,c in enumerate(ids)}

def _load_matrix_npz(path):
    z=np.load(path,allow_pickle=True)
    ids=z['case_id'].astype(str)
    out={}
    for key in z.files:
        if key=='case_id': continue
        out[key]={c:z[key][i] for i,c in enumerate(ids)}
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cohort',required=True,help='case_id,project_id,split + availability flags')
    ap.add_argument('--image',required=True,help='NPZ: case_id,features [N,512]')
    ap.add_argument('--text',required=True,help='NPZ: case_id,features [N,768]')
    ap.add_argument('--genomic',required=True,help='NPZ: case_id,features [N,G]')
    ap.add_argument('--labels',required=True,help='NPZ: case_id,labels [N,80]')
    ap.add_argument('--rules',required=True,help='NPZ: case_id,guide,target,contra,contra_indicator [N,80]')
    ap.add_argument('--out',default='data/model_inputs.npz')
    args=ap.parse_args()
    cohort=pd.read_csv(args.cohort)
    req={'case_id','project_id','split'}
    if not req.issubset(cohort.columns): raise ValueError(f'cohort missing {sorted(req-set(cohort.columns))}')
    image=_load_feature_npz(args.image); text=_load_feature_npz(args.text); genomic=_load_feature_npz(args.genomic)
    lab=_load_matrix_npz(args.labels); rules=_load_matrix_npz(args.rules)
    arrays={k:[] for k in ['image','text','genomic','mask','labels','guide','target','contra','contra_indicator','case_id','project_id','split']}
    for r in cohort.to_dict('records'):
        c=str(r['case_id'])
        # zero-fill missing modalities and preserve mask
        if c in image: im=np.asarray(image[c],dtype=np.float32); mi=1.0
        else: im=np.zeros(512,np.float32); mi=0.0
        if c in text: tx=np.asarray(text[c],dtype=np.float32); mt=1.0
        else: tx=np.zeros(768,np.float32); mt=0.0
        if c in genomic: ge=np.asarray(genomic[c],dtype=np.float32); mg=1.0
        else:
            # infer genomic dimensionality from first available row
            gd=len(next(iter(genomic.values())))
            ge=np.zeros(gd,np.float32); mg=0.0
        if c not in lab.get('labels',{}): raise ValueError(f'missing labels for {c}')
        for k in ['guide','target','contra','contra_indicator']:
            if c not in rules.get(k,{}): raise ValueError(f'missing {k} for {c}')
        arrays['image'].append(im); arrays['text'].append(tx); arrays['genomic'].append(ge); arrays['mask'].append([mi,mt,mg])
        arrays['labels'].append(lab['labels'][c]);
        for k in ['guide','target','contra','contra_indicator']: arrays[k].append(rules[k][c])
        arrays['case_id'].append(c); arrays['project_id'].append(str(r['project_id'])); arrays['split'].append(str(r['split']))
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.out,
        image=np.asarray(arrays['image'],np.float32), text=np.asarray(arrays['text'],np.float32),
        genomic=np.asarray(arrays['genomic'],np.float32), mask=np.asarray(arrays['mask'],np.float32),
        labels=np.asarray(arrays['labels'],np.float32), guide=np.asarray(arrays['guide'],np.float32),
        target=np.asarray(arrays['target'],np.float32), contra=np.asarray(arrays['contra'],np.float32),
        contra_indicator=np.asarray(arrays['contra_indicator'],np.float32),
        case_id=np.asarray(arrays['case_id'],object), project_id=np.asarray(arrays['project_id'],object), split=np.asarray(arrays['split'],object))
    print({'rows':len(cohort),'out':args.out})
if __name__=='__main__': main()
