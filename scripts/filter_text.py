from __future__ import annotations
import argparse
import pandas as pd
from knesynet.preprocessing.strict_text_filter import FrozenTreatmentLexicon, temporal_eligible, filter_for_prediction_time

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--documents',required=True,help='CSV with case_id,document_id,document_date,index_date,text')
    ap.add_argument('--lexicon',required=True)
    ap.add_argument('--out-documents',default='data/text_filtered.csv')
    ap.add_argument('--out-audit',default='data/document_level_text_filter_audit_private.csv')
    args=ap.parse_args()
    docs=pd.read_csv(args.documents)
    lex=FrozenTreatmentLexicon.from_csv(args.lexicon)
    filtered=[]; audit=[]
    for r in docs.to_dict('records'):
        eligible=temporal_eligible(r.get('document_date'),r.get('index_date'))
        if not eligible:
            audit.append({
                'case_id':r['case_id'],'document_id':r['document_id'],
                'temporally_eligible':0,'treatment_term_detected':0,'n_treatment_terms':0,
                'n_sentences_removed':0,'post_treatment_content_detected':0,'final_included':0,
            })
            continue
        text,info=filter_for_prediction_time(r.get('text',''),lex)
        include=int(bool(text))
        if include:
            filtered.append({'case_id':r['case_id'],'document_id':r['document_id'],'filtered_text':text})
        audit.append({
            'case_id':r['case_id'],'document_id':r['document_id'],'temporally_eligible':1,
            **info,'final_included':include,
        })
    pd.DataFrame(filtered).to_csv(args.out_documents,index=False)
    pd.DataFrame(audit).to_csv(args.out_audit,index=False)
    summary={
        'linked_documents':len(docs),
        'temporally_eligible':sum(x['temporally_eligible'] for x in audit),
        'removed_sentences':sum(x['n_sentences_removed'] for x in audit),
        'final_encoded_documents':len(filtered),
    }
    print(summary)

if __name__=='__main__': main()
