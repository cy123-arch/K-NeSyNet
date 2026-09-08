from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import torch
from knesynet.preprocessing.image_encoder import FrozenResNet50
from knesynet.preprocessing.wsi_pipeline import iter_svs_tiles

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True,help='case_id,svs_path'); ap.add_argument('--out',required=True); args=ap.parse_args()
    df=pd.read_csv(args.manifest); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); enc=FrozenResNet50().to(dev).eval()
    ids=[]; feats=[]
    for r in df.to_dict('records'):
        tiles=list(iter_svs_tiles(r['svs_path']))
        if not tiles: continue
        z=enc.encode_tiles(tiles,dev).mean(0).detach().cpu().numpy()
        # deterministic 2048->512 projection using first 512 principal-like coordinates is replaced in final model by learned projection
        # Here we retain the 2048-D source embedding for the downstream learned projector stage.
        ids.append(str(r['case_id'])); feats.append(z.astype(np.float32))
    np.savez_compressed(args.out,case_id=np.asarray(ids,object),features=np.asarray(feats,np.float32))
    print({'cases':len(ids),'feature_dim':int(feats[0].shape[0]) if feats else 0})
if __name__=='__main__': main()
