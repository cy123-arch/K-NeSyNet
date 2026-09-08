import argparse, json
from knesynet.evaluation.evaluate_outputs import evaluate_patient_output_csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patient_outputs")
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    print(json.dumps(evaluate_patient_output_csv(args.patient_outputs, args.threshold, args.k), indent=2))

if __name__ == "__main__":
    main()
