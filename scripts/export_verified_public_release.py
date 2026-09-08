#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, hmac, json, math, shutil
from pathlib import Path


def splitlist(x): return [v.strip() for v in str(x or '').split(';') if v.strip()]
def splitset(x): return set(splitlist(x))
def pseudo(secret: bytes, prefix: str, raw: str):
    return prefix + '-' + hmac.new(secret, str(raw).encode(), hashlib.sha256).hexdigest()[:18]

def patient_metrics(row):
    y=splitset(row['true_labels']); pred=splitset(row['thresholded_positive_drugs']); top=splitlist(row['top10_drugs'])[:10]
    tp=len(y & pred); fp=len(pred-y); fn=len(y-pred)
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    j=tp/(tp+fp+fn) if tp+fp+fn else 0.0
    hit=len(y & set(top)); p=hit/10.; r=hit/len(y) if y else 0.0; cf=2*p*r/(p+r) if p+r else 0.0
    rel=[1 if d in y else 0 for d in top]
    dcg=sum(v/math.log2(i+2) for i,v in enumerate(rel)); k=min(len(y),10)
    idcg=sum(1/math.log2(i+2) for i in range(k)) if k else 1.0
    return f1,j,p,r,cf,(dcg/idcg if k else 0.0)

def export_patients(src,dst,secret):
    with open(src,newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    req=['case_id','project_id','true_labels','top10_drugs','top10_scores','thresholded_positive_drugs','guideline_flags_top10','contraindication_flags_top10']
    miss=[x for x in req if not rows or x not in rows[0]]
    if miss: raise ValueError('patient input missing '+str(miss))
    if len(rows)!=718: raise ValueError(f'expected exactly 718 test rows, got {len(rows)}')
    out=[]; ms=[]
    for r in rows:
        m=patient_metrics(r); ms.append(m)
        out.append({
          'study_case_id':pseudo(secret,'KSN',r['case_id']), 'project_id':r['project_id'],
          'true_labels':r['true_labels'], 'top10_drugs':r['top10_drugs'], 'top10_scores':r['top10_scores'],
          'thresholded_positive_drugs':r['thresholded_positive_drugs'],
          'guideline_flags_top10':r['guideline_flags_top10'], 'contraindication_flags_top10':r['contraindication_flags_top10']})
    with open(dst,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    means=[sum(x[i] for x in ms)/len(ms) for i in range(6)]
    return {'n':len(out),'CR-F1@10':means[0],'Jaccard':means[1],'Precision@10':means[2],'Recall@10':means[3],'Conventional F1@10':means[4],'NDCG@10':means[5]}

def export_docs(src,dst,secret):
    with open(src,newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    req=['case_id','document_id','temporally_eligible','treatment_term_detected','n_treatment_terms','n_sentences_removed','post_treatment_content_detected','final_included']
    miss=[x for x in req if not rows or x not in rows[0]]
    if miss: raise ValueError('document audit missing '+str(miss))
    out=[]
    for r in rows:
        out.append({
          'study_case_id':pseudo(secret,'KSN',r['case_id']),
          'study_document_id':pseudo(secret,'DOC',r['document_id']),
          'temporally_eligible':r['temporally_eligible'],'treatment_term_detected':r['treatment_term_detected'],
          'n_treatment_terms':r['n_treatment_terms'],'n_sentences_removed':r['n_sentences_removed'],
          'post_treatment_content_detected':r['post_treatment_content_detected'],'final_included':r['final_included']})
    with open(dst,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    eligible=sum(int(float(r['temporally_eligible'])) for r in rows)
    removed=sum(int(float(r['n_sentences_removed'])) for r in rows)
    return {'n_linked_documents':len(rows),'n_temporally_eligible':eligible,'n_removed_sentences':removed}

def copy_lexicon(src,dst):
    shutil.copyfile(src,dst)
    with open(dst,'rb') as f: sha=hashlib.sha256(f.read()).hexdigest()
    with open(dst,newline='',encoding='utf-8-sig') as f: n=sum(1 for _ in csv.DictReader(f))
    return {'rows':n,'sha256':sha}

def sha256s(directory):
    lines=[]
    for p in sorted(directory.iterdir()):
        if p.is_file() and p.name!='SHA256SUMS.txt':
            h=hashlib.sha256(p.read_bytes()).hexdigest(); lines.append(f'{h}  {p.name}')
    (directory/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--patient-private',required=True)
    ap.add_argument('--document-private',required=True)
    ap.add_argument('--lexicon',required=True)
    ap.add_argument('--threshold-selection')
    ap.add_argument('--drug-vocabulary')
    ap.add_argument('--cohort-manifest')
    ap.add_argument('--secret',required=True,help='private HMAC secret; never publish')
    ap.add_argument('--outdir',default='VERIFIED_PUBLIC_RELEASE')
    ap.add_argument('--expected-cr-f1',type=float)
    ap.add_argument('--expected-jaccard',type=float)
    ap.add_argument('--tol',type=float,default=5e-4)
    args=ap.parse_args()
    out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    secret=args.secret.encode()
    pm=export_patients(args.patient_private,out/'patient_level_test_outputs.csv',secret)
    dm=export_docs(args.document_private,out/'document_level_text_filter_audit.csv',secret)
    lx=copy_lexicon(args.lexicon,out/'treatment_regimen_lexicon.csv')
    for src,name in [(args.threshold_selection,'threshold_selection.csv'),(args.drug_vocabulary,'drug_vocabulary.csv'),(args.cohort_manifest,'cohort_split_manifest.csv')]:
        if src: shutil.copyfile(src,out/name)
    checks=[]
    if args.expected_cr_f1 is not None:
        checks.append(('CR-F1@10',pm['CR-F1@10'],args.expected_cr_f1))
    if args.expected_jaccard is not None:
        checks.append(('Jaccard',pm['Jaccard'],args.expected_jaccard))
    for k,v,e in checks:
        if abs(v-e)>args.tol: raise ValueError(f'{k} recomputed={v:.8f} differs from expected={e:.8f}; do not publish until reconciled')
    report={'patient_metrics':pm,'document_audit':dm,'lexicon':lx}
    (out/'verification_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    sha256s(out)
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
