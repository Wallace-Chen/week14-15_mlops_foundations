# Financial MLOps Model Serving

![CI](https://github.com/Wallace-Chen/week14-15_mlops_foundations/actions/workflows/ci.yml/badge.svg)

## Overview

This Week 14-15 MLOps foundations project turns an earlier SPY financial ML feature pipeline into a small production-style model-serving system.

The goal is not maximum trading performance. The goal is to show that a research/notebook model can become a reproducible service with:

- explicit model artifacts and metadata
- deterministic feature validation and ordering
- FastAPI serving endpoints
- Dockerized runtime
- pytest coverage
- GitHub Actions CI
- structured monitoring logs, drift-monitoring plan, and system-design writeup

> A notebook is an experiment. A service is a contract.

## Repository Structure

```text
.
├── README.md
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
├── .github/workflows/ci.yml
├── configs/serving.yaml
├── data/sample_request.json
├── models/
│   ├── model.pkl
│   └── metadata.json
├── reports/
│   ├── api_examples.md
│   ├── ml_monitoring_plan.md
│   └── mlops_system_design.md
├── scripts/
│   ├── build_docker.sh
│   ├── run_local_api.py
│   ├── smoke_test_api.py
│   └── train_baseline_model.py
├── src/financial_mlops/
│   ├── features.py
│   ├── model.py
│   ├── monitoring.py
│   ├── schemas.py
│   └── service.py
└── tests/
    ├── test_api.py
    ├── test_features.py
    ├── test_model.py
    └── test_monitoring.py
```

## Architecture

```text
SPY feature data / sample request
        ↓
feature validation and metadata-defined ordering
        ↓
saved sklearn model artifact loaded once at startup
        ↓
FastAPI service: /health, /metadata, /predict
        ↓
prediction response with probability, model version, latency, request id
        ↓
structured JSON logs + optional local CSV audit logs
        ↓
monitoring plan for latency, errors, data drift, prediction drift, and model drift
```

## Model Artifact

The served model is a baseline SPY next-day direction classifier.

- Model name: `spy_direction_baseline`
- Version: `0.1.0`
- Target: `next_day_direction`
- Artifact: `models/model.pkl`
- Metadata: `models/metadata.json`
- Model type: `sklearn.ensemble.RandomForestClassifier`
- Training data source: `week3-4_Mar2026/data/SPY_features.csv`
- Feature contract: 28 ordered features stored in `models/metadata.json`

This is a serving and MLOps demo, not investment advice.

## API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check service readiness and loaded model version. |
| `GET` | `/metadata` | Return model metadata, feature contract, model type, and metrics. |
| `POST` | `/predict` | Validate feature payload and return prediction, probability, latency, and request id. |

Example prediction request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @data/sample_request.json
```

Example response shape:

```json
{
  "ticker": "SPY",
  "prediction": 0,
  "probability": 0.359,
  "model_version": "0.1.0",
  "latency_ms": 3.2,
  "request_id": "..."
}
```

More examples are in [`reports/api_examples.md`](reports/api_examples.md).

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/train_baseline_model.py
python -m pytest -q
```

Expected current test result:

```text
12 passed
```

## Run the API Locally

```bash
source .venv/bin/activate
python3 scripts/run_local_api.py
```

Or directly with Uvicorn:

```bash
PYTHONPATH=src uvicorn financial_mlops.service:app --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
python3 scripts/smoke_test_api.py
curl http://localhost:8000/health
curl http://localhost:8000/metadata
```

## Run with Docker

Build:

```bash
./scripts/build_docker.sh
# or
docker build -t financial-mlops-api:latest .
```

Run:

```bash
docker run --rm -p 8000:8000 financial-mlops-api:latest
```

Or with Docker Compose:

```bash
docker compose up --build
```

Confirm:

```bash
curl http://localhost:8000/health
python3 scripts/smoke_test_api.py
```

The serving image includes application code, dependencies, metadata, config, the approved model artifact, and sample request data. It intentionally excludes the original training CSV and notebooks.

## Tests

The test suite covers the core serving contract:

- `tests/test_features.py` — feature-vector length/order and clear validation errors.
- `tests/test_model.py` — artifact existence, metadata completeness, prediction output, probability range.
- `tests/test_api.py` — `/health`, `/metadata`, valid `/predict`, missing-feature rejection, malformed request rejection.
- `tests/test_monitoring.py` — feature-summary statistics and optional CSV logging behavior.

Run:

```bash
python -m pytest -q
```

## CI/CD

GitHub Actions workflow: [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

The workflow runs on push and pull request:

1. Check out the repository.
2. Set up Python 3.11 with pip caching.
3. Install dependencies and the editable local package.
4. Run `python -m pytest -q`.
5. Build the Docker image with tag `financial-mlops-api:test`.

This is CI rather than full CD. It checks tests, import/package health, API contract regressions, and Docker build readiness before deployment.

## Monitoring and Production Concerns

Current monitoring implementation:

- `src/financial_mlops/monitoring.py` emits structured JSON events.
- `src/financial_mlops/service.py` logs one prediction event for each prediction attempt.
- `reports/ml_monitoring_plan.md` defines production metrics, drift checks, alert thresholds, retraining triggers, and auditability requirements.

Prediction log fields include:

- `timestamp_utc`
- `request_id`
- `endpoint`
- `ticker`
- `model_version`
- compact feature summary: count, mean, std, min, max
- `prediction`
- `probability`
- `latency_ms`
- `error_status`
- `error_message`

Optional local CSV audit logging:

```bash
MLOPS_PREDICTION_LOG_CSV=logs/predictions.csv python3 scripts/run_local_api.py
python3 scripts/smoke_test_api.py
```

In production, these events should go to a real logging/metrics backend such as CloudWatch, GCP Logging, Datadog, Grafana Loki, Prometheus/OpenTelemetry, or an ML observability platform.

## System Design Report

The paper-style system design writeup is in [reports/mlops_system_design.md](reports/mlops_system_design.md). It covers architecture, training vs serving, batch vs online inference, feature consistency, reproducibility, containerization, testing/CI, monitoring, failure modes, security basics, cloud deployment, limitations, and rollback strategy.

## Limitations

- Baseline model quality is weak; this project prioritizes production structure over predictive performance.
- No authentication, authorization, or rate limiting yet.
- No model registry yet.
- No deployed cloud endpoint yet.
- Drift checks are documented but not yet automated as scheduled jobs.
- Prediction labels arrive with delay, so true performance monitoring requires a later label-join pipeline.

## Future Improvements

- Add `models/reference_stats.json` during training for automated feature-drift checks.
- Add Prometheus/OpenTelemetry middleware for request metrics.
- Add a scheduled monitoring script over `logs/predictions.csv` or a production log store.
- Add a model registry and versioned deployment/rollback workflow.
- Add cloud deployment to Cloud Run, ECS/Fargate, or a Kubernetes service.
- Add authentication and request-rate controls.

---

# Interview Talking Points

This section converts the study plan questions from `Phase3_Week14-15_MLOps_Foundations.md` into interview-ready answers.

## Week 14 Day 1-2: Project Skeleton + Model Artifact

### Why save metadata with the model artifact?

The model file stores learned parameters and estimator state, but it does not fully describe the serving contract. Metadata records the model name, version, target, feature order, training data, creation time, model type, metrics, and notes. Without metadata, another engineer can load the artifact but may not know what inputs it expects, which version is deployed, or whether it is safe to serve.

### Why should feature order be explicit?

Most tabular models consume numeric arrays, not dictionaries. If the same feature values are passed in a different order, the model can return a valid-looking but wrong prediction. Storing feature order in `models/metadata.json` and enforcing it in `features.py` makes inference deterministic, testable, and auditable.

### Why is notebook state dangerous for production ML?

Notebook state is implicit. Variables may depend on cells run earlier, cells run out of order, local paths, or objects that were never saved. Production services need repeatable startup from files on disk. This project separates training code, saved artifacts, metadata, and serving logic so the API can start cleanly without hidden notebook memory.

### Why separate training code from serving code?

Training code optimizes experimentation and artifact creation. Serving code enforces a stable runtime contract under latency and reliability constraints. Separating them keeps the API smaller, faster to start, easier to test, and less likely to accidentally retrain or mutate model state during inference.

## Week 14 Day 3-4: FastAPI Model Service

### Why should the model be loaded once at startup?

Loading the model once avoids repeated disk I/O and deserialization on every request. It reduces latency, makes behavior predictable, and exposes artifact-loading failures immediately when the service starts instead of failing only after the first user request.

### What is a health check?

A health check is a lightweight endpoint used by humans, scripts, and orchestrators to confirm the service is alive and ready. Here, `/health` returns service status, model-loaded status, and model version, so a deployment system can detect whether the API is ready for prediction traffic.

### What errors should `/predict` return for invalid input?

Invalid client payloads should return a 4xx error, not crash the server. This service returns HTTP 422 when required features are missing, unexpected features are present, or feature values cannot be converted to numeric inputs. True internal failures should be treated as 5xx errors.

### How would you monitor latency and prediction distribution?

Start with structured logs containing request id, ticker, model version, prediction, probability, feature summary, and latency. Aggregate those into p50/p95/p99 latency, error rate, request volume, prediction class balance, and probability-distribution dashboards. Sudden latency spikes indicate infrastructure problems; sudden prediction/probability shifts may indicate data or model drift.

## Week 14 Day 5-6: Dockerize the Service

### Why use containers for model serving?

Containers package application code, dependencies, model artifacts, metadata, config, and runtime command into one repeatable unit. That makes the service easier to run on another machine, in CI, or in cloud deployment because the runtime environment is defined by the image instead of by a developer laptop.

### What belongs inside the image vs mounted at runtime?

The image should include stable runtime assets: serving code, pinned dependencies, approved model artifact, metadata, config defaults, and a small sample request. Runtime-specific items such as secrets, environment-specific config, large logs, and frequently changing artifacts should be injected through environment variables, secret managers, volumes, or deployment tooling.

### Why avoid copying training data into the serving image?

Training data increases image size, slows builds and deployments, and can expose data the inference service does not need. A serving image should contain the minimum assets required to answer prediction requests. Training data belongs in the training pipeline or artifact store, not in the production API container.

### What makes a Docker image reproducible?

A reproducible image has an explicit base image, dependency file, deterministic build steps, and a clear runtime command. It avoids hidden host assumptions and excludes unnecessary local files with `.dockerignore`. For stricter reproducibility, pin package versions and use immutable base image digests.

## Week 15 Day 1-2: Unit Tests and API Tests

### How do you test ML code when model predictions can change?

Do not overfit tests to one exact predicted class unless the fixture and model artifact are intentionally frozen. For serving tests, focus on deterministic contracts: the model loads, the feature vector has the expected shape/order, the prediction field exists, the class is in the valid label set, and the probability is in `[0, 1]`. If exact outputs matter, pin the artifact, sample request, dependency versions, and random seeds, then treat it as a regression test.

### What should be deterministic in a serving pipeline?

The request schema, required feature names, feature ordering, validation behavior, model artifact version, response schema, and error semantics should be deterministic. For the same model artifact and same input payload, preprocessing and inference should produce the same feature vector and a valid repeatable response. Operational values such as request id and latency can vary, but their presence and types should still be tested.

### What is the difference between validating shape and validating performance?

Shape validation checks engineering correctness: the request has the right fields, features can be converted to numbers, the model receives the expected 2D array, and the API returns the promised fields. Performance validation checks model quality: accuracy, precision/recall, ROC-AUC, calibration, drift, or business metrics. A model can pass shape tests while performing poorly, so CI should catch contract regressions while offline evaluation and monitoring track predictive quality.

## Week 15 Day 3-4: GitHub Actions CI/CD Basics

### What should CI catch before deployment?

CI should catch broken imports, dependency problems, failing unit/API tests, schema regressions, missing artifacts, and Docker build failures. It should not guarantee model business value, but it should prevent obviously broken code or containers from moving toward deployment.

### Why build Docker in CI even if not deploying yet?

A Docker build is a strong deployment-readiness check. It confirms the Dockerfile, dependency installation, copied files, package imports, and runtime assumptions work in a clean environment. This catches many “works on my machine” issues before a real deploy step exists.

### What is the difference between unit tests and smoke tests?

Unit tests verify small pieces of logic, such as feature validation or model response shape. Smoke tests verify that the assembled system can perform a basic end-to-end action, such as starting the API and sending one valid prediction request. Unit tests diagnose specific failures; smoke tests catch major integration breakage quickly.

## Week 15 Day 5: Monitoring and Production Concerns

### What is data drift?

Data drift is a distribution shift in model inputs between the training/reference data and current production traffic. For this API, examples include return, volatility, volume, or technical-indicator distributions moving far away from the training baseline. Drift does not prove the model is wrong, but it warns that the model is operating outside familiar conditions.

### Why can model performance degrade without code changes?

The world can change while the code stays the same. Market regimes shift, volatility changes, liquidity changes, feature distributions move, and relationships learned during training can weaken or reverse. Data pipelines can also change upstream. This is why production ML needs monitoring of inputs, outputs, labels, and performance over time.

### What would you monitor for a financial prediction API?

I would monitor service metrics, data quality, prediction behavior, and delayed label-based performance. Concretely: request volume, latency, error rate, missing features, feature mean/std/ranges, feature drift vs training reference, prediction class distribution, probability distribution, calibration, accuracy/log loss after labels arrive, and all metrics segmented by ticker and market regime.

### Why is model versioning important for auditability?

Model versioning lets us connect every prediction to the exact artifact, metadata, feature contract, and training process that produced it. This is essential for debugging, rollback, compliance review, and post-event analysis. Without versioning, two predictions from different artifacts can look identical in logs even though they came from different models.

## Week 15 Day 6: Paper-Style System Design Writeup

### How would you deploy this on AWS/GCP?

On AWS, I would build the Docker image in CI, push it to ECR, and deploy it to ECS/Fargate or SageMaker real-time endpoints behind an Application Load Balancer or API Gateway. On GCP, I would push the image to Artifact Registry and deploy to Cloud Run for a simple autoscaling service. In both cases I would add secret management, structured logs, metrics, health checks, and a rollback path.

### How would you handle multiple model versions?

Use a model registry or artifact store with immutable versioned artifacts and metadata. Each deployed service should expose its model version through `/health`, `/metadata`, and prediction logs. For rollout, use blue/green or canary deployment: send a small traffic slice to the new version, compare metrics, then promote or roll back.

### How do you ensure training-serving feature consistency?

Use one shared feature-definition contract. Store the ordered feature list in metadata, validate every request against it, and test the feature vector shape/order in CI. For larger systems, use a feature store or shared transformation library so training and serving use the same definitions instead of duplicating logic.

### How would you roll back a bad model?

Keep previous model artifacts and container images available. If monitoring detects degraded performance, high error rates, or severe drift after rollout, redeploy the previous known-good image/model version. Because prediction logs include model version, we can identify when the bad model started, evaluate impact, and verify recovery after rollback.
