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


## FastAPI service (Day 3-4)

Run the API locally:

```bash
PYTHONPATH=src uvicorn financial_mlops.service:app --host 0.0.0.0 --port 8000
# or
python3 scripts/run_local_api.py
```

Validate the running service:

```bash
python3 scripts/smoke_test_api.py
curl http://localhost:8000/health
curl http://localhost:8000/metadata
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @data/sample_request.json
```

The service loads `models/model.pkl` and `models/metadata.json` once at startup, then uses the metadata feature list to validate and order every prediction request.

## Week 14 Interview Talking Points: Days 1-4

### Day 1-2: Project Skeleton + Model Artifact

**Why save metadata with the model artifact?**

The model file alone only stores learned parameters and estimator state. Metadata explains the serving contract around that artifact: model name, version, target, feature order, training data, creation date, model type, metrics, and notes. Without metadata, another engineer can load the model but may not know what inputs it expects or whether the artifact is safe to deploy.

**Why should feature order be explicit?**

Most tabular ML models consume numeric arrays, not dictionaries. If the same values are passed in a different order, the model can return a valid-looking but wrong prediction. Keeping feature order in `models/metadata.json` and using `features.py` to construct the vector makes inference deterministic and testable.

**Why is notebook state dangerous for production ML?**

Notebook state is implicit: variables may depend on cells run earlier, out of order, or not committed to source control. Production services need repeatable startup from files on disk. This project separates training code, saved artifacts, metadata, and inference logic so the service can start cleanly without relying on hidden notebook memory.

**Why separate training code from serving code?**

Training code optimizes experiments and artifact creation; serving code enforces a stable request/response contract under runtime constraints. Separating them keeps the API smaller, faster to start, easier to test, and less likely to accidentally retrain or mutate model state during inference.

### Day 3-4: FastAPI Model Service

**Why should the model be loaded once at startup?**

Loading the model once avoids repeated disk I/O and deserialization on every request. It reduces latency, makes behavior more predictable, and exposes artifact-loading failures immediately when the service starts instead of failing only after the first user request.

**What is a health check?**

A health check is a lightweight endpoint used by humans, scripts, or orchestrators to confirm the service is alive and ready. In this project, `/health` returns service status plus whether the model is loaded, so a deployment system can detect whether the API is ready to receive prediction traffic.

**What errors should `/predict` return for invalid input?**

Invalid client payloads should return a 4xx error, not a server crash. This service returns HTTP 422 when required features are missing, unexpected features are present, or feature values cannot be converted to numeric inputs. True internal failures are reserved for 5xx responses.

**How would you monitor latency and prediction distribution?**

Start with structured logs containing request id, ticker, prediction, probability, and latency. Over time, aggregate those logs into metrics: p50/p95/p99 latency, error rates, request volume, prediction class balance, probability distribution drift, and feature-value drift. Those signals help detect both infrastructure problems and model/data degradation.
