import argparse
from knesynet.preprocessing.labels import build_patient_labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treatments", required=True)
    ap.add_argument("--aliases", required=True)
    ap.add_argument("--vocabulary", required=True)
    ap.add_argument("--labels-out", default="data/patient_labels.csv")
    ap.add_argument("--audit-out", default="data/label_mapping_audit.csv")
    args = ap.parse_args()
    labels, audit = build_patient_labels(args.treatments, args.aliases, args.vocabulary)
    labels.to_csv(args.labels_out, index=False)
    audit.to_csv(args.audit_out, index=False)
    print(f"Wrote {len(labels)} patient label rows")

if __name__ == "__main__":
    main()
