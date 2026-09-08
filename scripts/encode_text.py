from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import torch
from knesynet.preprocessing.text_encoder import ClinicalBERTCaseEncoder

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--filtered-documents',required=True,help='case_id,document_id,filtered_text'); ap.add_argument('--out',required=True); ap.add_argument('--checkpoint',default='emilyalsentzer/Bio_ClinicalBERT'); ap.add_argument('--fine-tune',action='store_true'); args=ap.parse_args()
    df=pd.read_csv(args.filtered_documents); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc=ClinicalBERTCaseEncoder(checkpoint=args.checkpoint,fine_tune=args.fine_tune).to(dev)
    # For a feature-export stage, gradients are disabled; joint fine-tuning requires the raw-text training path used in the frozen experiment code.
    enc.eval(); ids=[]; feats=[]
    with torch.no_grad():
        for case_id,g in df.groupby('case_id',sort=False):
            z=enc.encode_documents(g['filtered_text'].astype(str).tolist(),dev)
            ids.append(str(case_id)); feats.append(z.cpu().numpy().astype(np.float32))
    np.savez_compressed(args.out,case_id=np.asarray(ids,object),features=np.asarray(feats,np.float32))
    print({'cases':len(ids),'dim':768})
if __name__=='__main__': main()
