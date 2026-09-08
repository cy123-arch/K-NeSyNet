# K-NeSyNet

K-NeSyNet is a multimodal neuro-symbolic framework for retrospective oncology treatment-record prediction. This repository contains the implementation and experiment utilities accompanying the manuscript.

## Repository contents

- multimodal preprocessing for pathology images, clinical text, and somatic mutation profiles;
- mask-aware patient representation learning;
- MM-OKG graph construction and treatment-edge leakage controls;
- graph-attention neural scoring;
- deterministic guideline, mutation-target, and contraindication channels;
- adaptive neural-symbolic gated fusion and safety-aware training;
- simple control baselines;
- evaluation metrics and patient-level export utilities;
- graph, text, metric, and score-decomposition audit tools.

## Environment

The manuscript implementation uses:

- Python 3.10.14
- PyTorch 2.3.1
- PyTorch Geometric 2.5.3
- Transformers 4.41.2
- NumPy 1.26.4
- pandas 2.2.2
- scikit-learn 1.5.0
- CUDA 12.1 / cuDNN 8.9.7

A single NVIDIA GeForce RTX 4090 (24 GB) is sufficient for the reported configuration.

Install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Data layout

Raw TCGA/GDC files and license-restricted third-party knowledge resources are not bundled in the repository. Expected input tables and schemas are documented under `data/templates/`.

The configured TCGA projects are:

`TCGA-BRCA, TCGA-LUAD, TCGA-LUSC, TCGA-COAD, TCGA-PRAD, TCGA-KIRC, TCGA-THCA, TCGA-HNSC, TCGA-STAD, TCGA-UCEC`.

The pipeline expects a frozen 80-drug candidate vocabulary, a drug-alias table, a 10,586-gene vocabulary, graph relation tables, and guideline/contraindication rule tables.

## Main workflow

1. Build and split the cohort.
2. Normalize treatment labels.
3. Preprocess imaging, text, and genomic inputs.
4. Build MM-OKG and apply graph split/leakage controls.
5. Train K-NeSyNet.
6. Evaluate the held-out test set.
7. Export patient-level ranked predictions and score decompositions.

Typical commands:

```bash
python scripts/build_labels.py --treatments data/treatments.csv --aliases data/drug_aliases.csv --vocabulary data/drug_vocabulary.csv
python scripts/build_cohort.py --config configs/final.yaml --case-manifest data/case_manifest.csv --labels data/patient_labels.csv
python scripts/build_graph.py --entities data/kg_entities.csv --edges data/kg_edges.csv --out data/graph.pt
python train.py --config configs/final.yaml --inputs data/model_inputs.npz --graph data/graph.pt
python evaluate.py --config configs/final.yaml --inputs data/model_inputs.npz --graph data/graph.pt --checkpoint checkpoints/best.pt
python scripts/export_patient_outputs.py --config configs/final.yaml --checkpoint checkpoints/best.pt
```

## Evaluation terminology

The implementation exposes both:

- candidate-restricted thresholded multilabel F1 after top-K restriction; and
- conventional Precision@K / Recall@K / F1@K computed from the full retrieved top-K set.

Jaccard is computed from the same binarized prediction matrix as the candidate-restricted thresholded F1.

## Third-party resources

TCGA/GDC data must be obtained from the NCI Genomic Data Commons under the applicable access conditions. DGIdb, KEGG, NCCN-derived knowledge, Bio_ClinicalBERT, ImageNet-pretrained ResNet-50, and other third-party resources remain subject to their own licenses and terms of use.

## Citation

See `CITATION.cff`.


## v3 integration additions

This package adds strict temporal-before-text filtering, an explicit `build_model_inputs.py` integration step, a global-graph training contract, threshold-selection export, release-data validation, a full command map, and an executable smoke test. See `PIPELINE_COMMANDS.md` and `CODE_METHOD_ALIGNMENT.md`.
