from __future__ import annotations
import argparse
import numpy as np
from knesynet.preprocessing.genomics import maf_to_multihot

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--maf',required=True); ap.add_argument('--gene-vocabulary',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
    ids,x=maf_to_multihot(args.maf,args.gene_vocabulary)
    np.savez_compressed(args.out,case_id=np.asarray(ids,object),features=x.astype(np.float32))
    print({'cases':len(ids),'genes':x.shape[1]})
if __name__=='__main__': main()
