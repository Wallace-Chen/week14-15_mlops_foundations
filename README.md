# Financial MLOps Model Serving

Week 14-15 / Day 1-2 deliverable: **Project Skeleton + Model Artifact** for Docker + Model Serving Foundations.

This project turns the earlier financial ML SPY feature dataset into a clean, serving-ready model package. The focus is not maximum trading performance; it is reproducible inference, explicit model metadata, and a stable feature contract.

## Day 1-2 outputs

- `models/model.pkl` — trusted local scikit-learn model artifact
- `models/metadata.json` — model version, target, feature order, training info, metrics
- `src/financial_mlops/features.py` — validates request keys and converts JSON to ordered feature vectors
- `src/financial_mlops/model.py` — loads artifact + metadata and exposes prediction helpers
- `data/sample_request.json` — deterministic sample input for smoke tests / future API tests

## Quick start

```bash
python3 scripts/train_baseline_model.py
python3 -m unittest discover -s tests -v
```

## Architecture sketch

```text
SPY_features.csv
   │
   ├── scripts/train_baseline_model.py
   │      ├── trains baseline classifier for next-day SPY direction
   │      ├── saves models/model.pkl
   │      └── writes models/metadata.json + data/sample_request.json
   │
   └── src/financial_mlops/
          ├── features.py  # request validation + feature ordering
          └── model.py     # artifact loading + inference contract
```

## Why this structure matters

- The model artifact is independent of notebook state.
- Metadata records the exact feature order required at inference time.
- Feature validation happens before model prediction.
- Training code and serving/inference code are separated.
