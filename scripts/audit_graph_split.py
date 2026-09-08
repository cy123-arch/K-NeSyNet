import argparse, pandas as pd
from knesynet.kg.splits import describe_patient_presence

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", required=True)
    ap.add_argument("--cohort", required=True)
    args = ap.parse_args()
    e = pd.read_csv(args.edges)
    c = pd.read_csv(args.cohort)
    for split in ["train","validation","test"]:
        ids = c.loc[c["split"].eq(split), "case_id"].astype(str).tolist()
        print(split, describe_patient_presence(e, ids))

if __name__ == "__main__":
    main()
