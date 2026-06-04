# Financial MLOps Model Serving

Week 14-15 / Day 1-2 deliverable: **Project Skeleton + Model Artifact** for Docker + Model Serving Foundations.

This project turns the earlier financial ML SPY feature dataset into a clean, serving-ready model package. The focus is not maximum trading performance; it is reproducible inference, explicit model metadata, and a stable feature contract.

## Day 1-2 outputs

- `models/model.pkl` — trusted local scikit-learn model artifact
- `models/metadata.json` — model version, target, feature order, training info, metrics
- `src/financial_mlops/features.py` — validates request keys and converts JSON to ordered feature vectors
- `src/financial_mlops/model.py` — loads artifact + metadata and exposes prediction helpers
- `data/sample_request.json` — deterministic sample input for smoke tests / future API tests


## Week 15 Day 1-2 outputs: Unit Tests and API Tests

The Week 15 testing pass adds pytest-based coverage for the model-serving contract:

- `tests/test_features.py` — verifies feature vector length/order and clear feature validation errors.
- `tests/test_model.py` — verifies saved artifacts, metadata completeness, prediction output, and probability range.
- `tests/test_api.py` — verifies `/health`, `/metadata`, valid `/predict`, missing-feature rejection, and malformed request rejection.
- `requirements.txt` — includes `pytest` and `httpx` so FastAPI `TestClient` works in local and CI-style environments.
- `pyproject.toml` — configures pytest to discover tests and import the `src/` package without manual `PYTHONPATH` hacks.

Run the test suite:

```bash
pytest -q
```

Expected result after this pass:

```text
10 passed
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/train_baseline_model.py
pytest -q
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


## Dockerized service (Day 5-6)

Build the image:

```bash
./scripts/build_docker.sh
# or
docker build -t financial-mlops-api:latest .
```

Run the container:

```bash
docker run --rm -p 8000:8000 financial-mlops-api:latest
```

Or use Docker Compose:

```bash
docker compose up --build
```

Confirm the containerized API:

```bash
curl http://localhost:8000/health
python3 scripts/smoke_test_api.py
```

The image contains only the serving package, model artifact, metadata, config, sample request, and documentation needed at runtime. The original training CSV and notebooks are intentionally not copied into the serving image.

## Week 14 Interview Talking Points: Days 1-6

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

### Day 5-6: Dockerize the Service

**Why use containers for model serving?**

Containers package the API code, Python dependencies, model artifact, metadata, and runtime command into one repeatable unit. That makes the service easier to run on another machine, in CI, or later in cloud deployment because the environment is defined by the image instead of by a developer laptop.

**What belongs inside the image vs mounted at runtime?**

The image should include stable runtime assets: application code, pinned dependencies, the approved model artifact, metadata, config defaults, and a small sample request. Runtime-specific items such as secrets, environment-specific config, large logs, and frequently changing artifacts should be injected through environment variables, secret managers, volumes, or deployment tooling.

**Why avoid copying training data into the serving image?**

Training data increases image size, slows builds and deployments, and can expose data that the inference service does not need. A serving image should have the minimum assets required to answer prediction requests. Training data belongs in the training pipeline or artifact store, not in the production API container.

**What makes a Docker image reproducible?**

A reproducible image has an explicit base image, dependency file, deterministic build steps, and a clear runtime command. It avoids hidden host assumptions and excludes unnecessary local files with `.dockerignore`. For stricter production reproducibility, pin package versions and use immutable base image digests.

## Week 15 Interview Talking Points: Day 1-2 Testing

### How do you test ML code when model predictions can change?

Do not overfit tests to one exact predicted class unless the fixture and model artifact are intentionally frozen. For serving tests, focus on deterministic contracts: the model loads, the feature vector has the expected shape/order, the prediction field exists, the class is in the valid label set, and the probability is in `[0, 1]`. If exact outputs matter, pin the model artifact, sample request, dependency versions, and random seeds, then treat that as a golden-file or regression test.

### What should be deterministic in a serving pipeline?

The request schema, required feature names, feature ordering, validation behavior, model artifact version, response schema, and error semantics should be deterministic. For the same model artifact and same input payload, preprocessing and inference should produce the same shaped feature vector and a valid, repeatable response. Operational values such as request id and latency can vary, but their presence and types should still be tested.

### What is the difference between validating shape and validating performance?

Shape validation checks engineering correctness: does the request contain the right features, can they be converted to numbers, does the model receive the expected 2D array, and does the API return the promised fields? Performance validation checks model quality: accuracy, precision/recall, ROC-AUC, calibration, drift, or business metrics. A model can pass shape tests while performing poorly, so CI should catch contract regressions while offline evaluation and monitoring track predictive quality.

