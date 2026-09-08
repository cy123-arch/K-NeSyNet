from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--npz',required=True,help='NPZ containing y_true [N,80], scores [N,80]')
    ap.add_argument('--out',default='outputs/threshold_selection.csv')
    args=ap.parse_args()
    z=np.load(args.npz)
    y=z['y_true']; scores=z['scores']
    grid=np.round(np.arange(0.05,0.951,0.01),2)
    vals=[]
    for t in grid:
        pred=(scores>=t).astype(int)
        vals.append(f1_score(y,pred,average='samples',zero_division=0))
    best=int(np.argmax(vals))
    df=pd.DataFrame({'threshold':grid,'validation_samples_f1':vals,'selected':[int(i==best) for i in range(len(grid))]})
    df.to_csv(args.out,index=False)
    print({'selected_threshold':float(grid[best]),'objective':'sample-averaged multilabel F1 over all candidate drugs','grid':'0.05..0.95 step 0.01'})
if __name__=='__main__': main()
