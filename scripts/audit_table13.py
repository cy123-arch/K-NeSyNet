import argparse
from knesynet.audit.fusion_equation import audit_table

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case_csv")
    ap.add_argument("--tolerance", type=float, default=1e-6)
    args = ap.parse_args()
    out = audit_table(args.case_csv, args.tolerance)
    print(out.to_string(index=False))
    if not out["pass"].all():
        raise SystemExit(2)

if __name__ == "__main__":
    main()
