from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile, numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader
from knesynet.data import ModelInputDataset
from knesynet.training.trainer import train_epoch
from knesynet.training.inference import collect_scores
from knesynet.training.losses import composite_loss
from knesynet.evaluation.thresholds import select_threshold
from knesynet.evaluation.metrics import cr_thresholded_f1, candidate_restricted_threshold_predictions, sample_jaccard

class SmokeModel(nn.Module):
    """Tiny model with the same forward contract as KNeSyNet, used only for integration testing."""
    def __init__(self, gene_count=10, n_drugs=5):
        super().__init__(); self.fc=nn.Linear(512+768+gene_count,n_drugs); self.gate=nn.Parameter(torch.zeros(n_drugs))
    def forward(self,image,text,genomic,mask,edge_index,drug_node_indices,guide,target,contra):
        neural=torch.sigmoid(self.fc(torch.cat([image,text,genomic],1)))
        symbolic=guide+target+contra; gate=torch.sigmoid(self.gate)[None,:].expand_as(neural)
        score=gate*neural+(1-gate)*torch.sigmoid(symbolic)
        return {'score':score,'neural':neural,'symbolic':symbolic,'gate':gate,'modality_weights':torch.full((image.shape[0],3),1/3,device=image.device)}

def main():
    rng=np.random.default_rng(7); n=12; nd=5; g=10
    split=np.array(['train']*8+['validation']*2+['test']*2,object)
    labels=(rng.random((n,nd))>0.7).astype(np.float32); labels[labels.sum(1)==0,0]=1
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'model_inputs.npz'
        np.savez_compressed(path,
            image=rng.normal(size=(n,512)).astype(np.float32),text=rng.normal(size=(n,768)).astype(np.float32),
            genomic=rng.normal(size=(n,g)).astype(np.float32),mask=np.ones((n,3),np.float32),labels=labels,
            guide=(rng.random((n,nd))>0.5).astype(np.float32),target=np.zeros((n,nd),np.float32),contra=np.zeros((n,nd),np.float32),
            contra_indicator=np.zeros((n,nd),np.float32),case_id=np.array([f'S{i:02d}' for i in range(n)],object),
            project_id=np.array(['SMOKE']*n,object),split=split)
        tr=DataLoader(ModelInputDataset(path,'train'),batch_size=4,shuffle=False)
        va=DataLoader(ModelInputDataset(path,'validation'),batch_size=2,shuffle=False)
        model=SmokeModel(g,nd); opt=torch.optim.Adam(model.parameters(),lr=1e-3); dev=torch.device('cpu')
        edge_index=torch.tensor([[0,1],[1,0]],dtype=torch.long); drug_nodes=torch.arange(nd,dtype=torch.long)
        def lf(scores,y,c,m): return composite_loss(scores,y,c,m,alpha=0,beta=0)
        loss=train_epoch(model,tr,opt,lf,dev,edge_index,drug_nodes)
        out=collect_scores(model,va,edge_index,drug_nodes,dev); t,_=select_threshold(out['labels'],out['scores'])
        f=cr_thresholded_f1(out['labels'],out['scores'],min(3,nd),t)
        pred=candidate_restricted_threshold_predictions(out['scores'],min(3,nd),t); j=sample_jaccard(out['labels'],pred)
        assert np.isfinite(loss) and np.isfinite(f) and np.isfinite(j)
        print({'smoke':'PASS','train_loss':float(loss),'threshold':float(t),'CR_F1':float(f),'Jaccard':float(j)})
if __name__=='__main__': main()
