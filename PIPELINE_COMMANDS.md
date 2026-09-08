# Full-data pipeline order

The public repository does not redistribute controlled TCGA/GDC source data or patient-linked source text. After obtaining the authorized source files, run the pipeline in this order and with the exact frozen data manifests/configuration used for the experiment.

1. Build/verify cohort and recorded-treatment labels
2. Apply temporal text exclusion and treatment-content filtering **before** ClinicalBERT input creation
3. Encode/prepare image, text, and genomic modalities
4. Build rule matrices and the split-dependent graph; remove held-out `treated_with` edges before message passing
5. Assemble `data/model_inputs.npz`
6. Train using global `edge_index` and `drug_node_indices`
7. Select the validation threshold on the 0.05--0.95 grid (step 0.01) under the frozen objective
8. Evaluate and export held-out patient outputs
9. Run audit utilities and verify file hashes

Representative commands:

```bash
python scripts/filter_text.py --documents PRIVATE/documents.csv --lexicon data/treatment_regimen_lexicon.csv --out-documents data/text_filtered.csv --out-audit PRIVATE/document_audit.csv
python scripts/encode_text.py --filtered-documents data/text_filtered.csv --out data/text_features.npz
python scripts/encode_wsi.py --manifest PRIVATE/image_manifest.csv --out data/image_features_2048.npz
python scripts/encode_genomics.py --maf PRIVATE/mutations.maf.tsv --gene-vocabulary data/gene_vocabulary.csv --out data/genomic_features.npz
python scripts/build_graph.py --entities data/kg_entities.csv --edges data/kg_edges.csv --out data/graph.pt
python scripts/build_model_inputs.py --cohort data/cohort.csv --image data/image_features_512.npz --text data/text_features.npz --genomic data/genomic_features.npz --labels data/labels.npz --rules data/rules.npz --out data/model_inputs.npz
python train.py --config configs/final.yaml --inputs data/model_inputs.npz --graph data/graph.pt --checkpoint checkpoints/best.pt
python evaluate.py --config configs/final.yaml --inputs data/model_inputs.npz --graph data/graph.pt --checkpoint checkpoints/best.pt
```

Use `python scripts/run_pipeline.py --smoke-test` to exercise the shared dataset/training/inference/metric contracts without external downloads.
