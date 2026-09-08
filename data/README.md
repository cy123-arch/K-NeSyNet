# Data interfaces

This directory holds study-specific tabular assets and derived inputs.

Expected files:

- `drug_vocabulary.csv`: frozen 80-drug candidate vocabulary.
- `drug_aliases.csv`: alias/brand/regimen-term normalization table.
- `gene_vocabulary.csv`: frozen 10,586-gene vocabulary.
- `kg_entities.csv`, `kg_edges.csv`: MM-OKG entity and relation tables.
- `guideline_rules.csv`: encoded positive guideline rules.
- `contraindication_rules.csv`: encoded contraindication/avoidance rules.
- `drug_targets.csv`: drug-gene target relations.
- `cohort.csv`: patient split and modality availability.
- `patient_labels.csv`: patient-level recorded-treatment labels.

Schemas are provided under `data/templates/`.

TCGA/GDC source files and license-restricted third-party source content are not redistributed here.
