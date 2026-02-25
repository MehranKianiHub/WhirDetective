# WhirDetective

WhirDetective is a Python project for industrial bearing fault diagnostics. It combines data engineering, feature projection, baseline ML training, calibration, KPI gating, and deployment packaging for controlled runtime environments.

## What It Does

- Builds canonical training datasets from raw bearing data (CWRU, Paderborn).
- Uses sliding-window segmentation and deterministic feature projection.
- Trains and evaluates a baseline 1D CNN for fault classification.
- Calibrates confidence and supports abstention-aware evaluation.
- Enforces KPI and release gates before deployment.
- Produces auditable artifacts (model card, KPI report, run report, manifests).
- Supports required benchmark sign-off including prognostics (XJTU-SY baseline).
- Supports runtime freeze, EdgeOS export, TFLite conversion, and staging canary prep.

## Core Features

- Data pipeline:
  - Group-aware train/val/test splitting to reduce leakage.
  - Windowed sample generation from multi-channel sensor signals.
  - Dataset fingerprinting for traceability.
- Feature projection:
  - Time-domain and spectral features (mean, std, RMS, peak-to-peak, abs max, skewness, kurtosis, crest factor, impulse factor, spectral centroid, spectral entropy, spectral flatness).
- Model and training:
  - `BaselineBearingCNN` with three convolutional blocks.
  - Balanced sampling option for class imbalance.
  - Temperature scaling and abstention threshold tuning.
- Safety and release controls:
  - KPI gate (accuracy, macro recall, calibration/ECE, coverage, selective accuracy).
  - Release gate (model size, parameter count, latency, integrity checks).
  - Package verification with checksums and optional signature verification.
- Deployment and operations:
  - Controlled runtime qualification freeze.
  - EdgeOS deployable package export.
  - Staging canary preparation with runtime smoke checks and rollback-oriented outputs.

## Documentation And Wiki

- Project wiki: https://github.com/MehranKianiHub/WhirDetective/wiki
- Wiki home (best starting point): https://github.com/MehranKianiHub/WhirDetective/wiki/Home
- Wiki pages index: https://github.com/MehranKianiHub/WhirDetective/wiki/_pages

## Datasets

Dataset discovery index (required):
- https://github.com/VictorBauler/awesome-bearing-dataset

WhirDetective currently includes workflows around these local dataset roots:
- `data/raw/cwru/`
- `data/raw/paderborn/`
- `data/raw/xjtu_sy/`
- `data/raw/ims/`
- `data/raw/femto_st/`

Notes:
- The awesome-bearing-dataset repository is an index, not a single dataset download.
- Organize raw files under `data/raw/...` and generated outputs under `data/processed/` or `artifacts/`.

## Installation

### Prerequisites

- Python 3.11+
- Recommended: virtual environment
- For full local development: optional dependencies in `.[dev,data]`

### Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,data]"
```

If you only need runtime usage (no dev tooling extras):

```bash
python -m pip install -e .
```

## How To Use

### CLI Command Reference

| Command | Purpose | Typical Inputs | Primary Outputs |
| --- | --- | --- | --- |
| `whirdetective-runner` | Run Step 4 diagnosis workflow (build dataset, train, calibrate, evaluate, package). | `--dataset-name`, `--dataset-root`, `--output-dir` | `model_card.json`, `kpi_report.json`, `run_report.json`, `release_gate.json`, model package files |
| `whirdetective-signoff` | Run required benchmark sign-off across diagnosis and prognostics tracks. | `--workspace-root`, `--output-dir` | `required_signoff.json` plus per-track artifacts |
| `whirdetective-runtime-freeze` | Execute runtime qualification checks and freeze a release-candidate bundle. | `--pilot-root`, `--freeze-root`, optional runtime suite options | `pilot_release_signoff.json`, `PILOT_RELEASE_SIGNOFF.md`, `freeze_manifest.json`, frozen package copy |
| `whirdetective-tflite-convert` | Convert a trained PyTorch baseline package to TFLite and run parity checks. | `--source-package-dir`, `--output-model` | `*.tflite`, `*.conversion_report.json` |
| `whirdetective-edgeos-export` | Build controlled EdgeOS deployable package from frozen artifacts and model blob. | `--source-package-dir`, `--model-blob`, `--output-dir` | `edgeos_export_report.json`, deployable package manifests/artifacts |
| `whirdetective-staging-canary` | Validate freeze/package integrity and prepare staged canary rollout inputs. | `--freeze-dir`, `--track`, optional runtime test options | `staging_canary_report.json`, rollout helper commands/logs |

Tip: run `<command> --help` for the full argument list and defaults.

### 1. Run baseline diagnosis workflow

```bash
whirdetective-runner \
  --workspace-root . \
  --dataset-name cwru \
  --dataset-root data/raw/cwru \
  --output-dir artifacts/cwru \
  --epochs 12 \
  --batch-size 32 \
  --learning-rate 1e-3 \
  --fail-on-kpi \
  --fail-on-release-gate
```

Main outputs:
- `model_card.json`
- `kpi_report.json`
- `run_report.json`
- `release_gate.json`
- package files (`model_state_dict.pt`, `inference_config.json`, `calibration.json`, `edgeos_model_manifest.json`, `manifest.json`, optional `manifest.sig`)

### 2. Run required sign-off (diagnosis + prognostics)

```bash
whirdetective-signoff \
  --workspace-root . \
  --output-dir artifacts/required-signoff
```

Main output:
- `artifacts/required-signoff/required_signoff.json`

### 3. Run runtime qualification freeze

```bash
whirdetective-runtime-freeze \
  --workspace-root . \
  --pilot-root artifacts/pilot \
  --freeze-root artifacts/pilot/freeze
```

### 4. Convert trained package to TFLite

```bash
whirdetective-tflite-convert \
  --workspace-root . \
  --source-package-dir artifacts/cwru \
  --output-model artifacts/edge/model.tflite
```

### 5. Export EdgeOS deployable package

```bash
whirdetective-edgeos-export \
  --workspace-root . \
  --source-package-dir artifacts/cwru \
  --model-blob artifacts/edge/model.tflite \
  --output-dir artifacts/edge/export \
  --fail-on-gate
```

### 6. Prepare staging canary inputs

```bash
whirdetective-staging-canary \
  --workspace-root . \
  --freeze-dir artifacts/pilot/freeze/<freeze-label> \
  --track cwru
```

## Development Workflow

Run quality checks:

```bash
pytest
ruff check .
mypy src
```

Helpful source areas:
- `src/whirdetective/data/` for ingestion, transforms, splitting, adapters.
- `src/whirdetective/ml/` for models and feature projection.
- `src/whirdetective/training/` and `src/whirdetective/evaluation/` for workflow and KPIs.
- `src/whirdetective/cli/` for operational command entrypoints.
- `src/whirdetective/export/` and `schemas/` for packaging and manifest contracts.

## How To Extend The Project

### Add another ML model

1. Implement a new model class in `src/whirdetective/ml/`.
2. Add model selection wiring in training workflow (`src/whirdetective/training/` and/or CLI args).
3. Ensure model metadata is captured in model card and package manifest outputs.
4. Add tests under `tests/unit/ml/` and training workflow tests.
5. Validate gate impact (accuracy, calibration, latency, model size).

### Add another CLI command

1. Create `src/whirdetective/cli/<new_command>.py` with `build_parser()` and `main()`.
2. Reuse existing parser/workflow patterns from `runner.py`, `signoff.py`, or `edgeos_export.py`.
3. Register the command in `pyproject.toml` under `[project.scripts]`.
4. Add CLI tests in `tests/unit/cli/`.
5. Emit JSON artifacts with stable keys for downstream automation.

### Extend security and release controls

1. Add new checks in KPI or release gate evaluation logic.
2. Keep integrity evidence in generated manifests and reports.
3. Enforce fail-fast behavior via CLI flags (for CI/CD gating).
4. Add schema validation where new JSON contracts are introduced.
5. Add runtime qualification tests before freeze/canary stages.

## Security Model

WhirDetective treats ML as advisory inside deterministic operational controls.

- Deterministic safety boundaries remain explicit.
- KPI and release gates block weak or non-compliant artifacts.
- Manifest checksums and optional signatures protect artifact integrity.
- Runtime qualification and canary flows reduce deployment risk.
- Gate failures can return non-zero exit codes for pipeline enforcement.

For signing, the default environment variable is:
- `WHIRDETECTIVE_MANIFEST_SIGNING_KEY`

## 📞 Contact

- 🌐 [Website](https://bootctrl.com/)
- 📧 [Email](mailto:mehran.kiani@bootctrl.com)
- 💬 [Discussions](https://github.com/MehranKianiHub/BootCtrl-EdgeOS/discussions)
- 🐛 [Issues](https://github.com/MehranKianiHub/BootCtrl-EdgeOS/issues)

<p align="center">
  <strong>⭐ Star us on GitHub if you find this project useful! ⭐</strong>
</p>

<p align="center">
  Made with ❤️ by the industrial automation and ML community
</p>

---