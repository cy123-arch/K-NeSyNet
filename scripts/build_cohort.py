import argparse
from knesynet.config import load_config
from knesynet.preprocessing.cohort import build_cohort

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/final.yaml")
    ap.add_argument("--case-manifest", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="data/cohort.csv")
    args = ap.parse_args()
    cfg = load_config(args.config)
    df = build_cohort(args.case_manifest, cfg["data"]["projects"], args.labels)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} cohort rows to {args.out}")

if __name__ == "__main__":
    main()
