# Code--Methods alignment checklist

Before public release, compare this integration package against the frozen experiment repository and confirm each item:

- Clinical text mode: joint Bio_ClinicalBERT fine-tuning vs fixed/precomputed 768-D embeddings.
- Exact SVS tissue detection, Macenko parameters, 20x-equivalent level selection, tile count, and 2048->512 projection.
- Exact gene vocabulary ordering and 10,586->256 genomic projection path.
- Exact guideline molecular-context conditions and contraindication/resistance conditions.
- Exact reverse-edge naming and graph construction order.
- Exact validation threshold objective, grid, tie-breaking, and selected threshold.
- Exact baseline repositories/versions/adapters.

Do not claim that a code path was used in the reported experiment until it has been compared with the frozen experiment code/configuration.
