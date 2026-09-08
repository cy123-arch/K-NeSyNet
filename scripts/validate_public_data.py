from __future__ import annotations
import argparse, pandas as pd

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--patient-outputs'); ap.add_argument('--document-audit'); ap.add_argument('--lexicon'); args=ap.parse_args()
    if args.patient_outputs:
        p=pd.read_csv(args.patient_outputs); assert len(p)==718, f'expected 718 rows, got {len(p)}'
        req={'study_case_id','true_labels','top10_drugs','top10_scores','thresholded_positive_drugs','guideline_flags_top10','contraindication_flags_top10'}
        assert req.issubset(p.columns), req-set(p.columns)
    if args.document_audit:
        d=pd.read_csv(args.document_audit); req={'study_case_id','study_document_id','temporally_eligible','treatment_term_detected','n_sentences_removed','final_included'}; assert req.issubset(d.columns), req-set(d.columns)
    if args.lexicon:
        l=pd.read_csv(args.lexicon); assert len(l)>0
    print('public-data validation PASS')
if __name__=='__main__': main()
