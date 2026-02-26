# Open Source Scope

This document defines what is intentionally included/excluded in `clean_for_opensource_final`.

## Included (public release)

- Core training/inference/evaluation entry points:
  - `train.py`
  - `generate.py`
  - `eval_simple.py`
- Core implementation packages:
  - `src/`
  - `data_utils/` (minimal runtime subset only: `PrepareDataset.py`, `MiniTools.py`, `__init__.py`)
- External runtime dependency:
  - `flow_matching` from `https://github.com/facebookresearch/flow_matching` (installed via `requirements.txt`)
- Public configs:
  - `src/config/config.yaml`
  - `src/config/config_chengdu.yaml`
  - `src/config/config_xian.yaml`
  - `config/config.yaml`
- Project metadata/docs:
  - `README.md`, `requirements.txt`, `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`, `RELEASE.md`

## Excluded (not part of public release)

- Private data references and private-region configurations.
- Benchmark/test harness for runtime validation only (batch runners and sweep configs).
- Generated artifacts and local outputs:
  - training outputs, checkpoints, generated CSVs/images, logs, analysis dumps.
- Raw data dumps and preprocessing pipelines that are not required for public-runtime usage.
- Large binary assets unsuitable for GitHub source distribution.

## Explicitly Removed from this final package

- Batch experiment and manager code under `src/tools/`.
- Batch/sweep config groups under `src/config/batch_configs/`, `src/config/opensource_1m/`, `src/config/baseline_exp/`.
- Non-essential eval batch scripts:
  - `src/eval/evaluate_meter.py`
  - `src/eval/training_manager.py`
  - `src/eval/generation_manager.py`
  - summary helper scripts not required by `eval_simple.py`
- Smoke/full-test scripts and run artifacts.
- Private-path shell scripts and conflicted-copy files.
- Unused legacy files under the old utilities package (e.g., legacy `TrajUNet*`, `EMA.py`, `logger.py`, and unused GIS json assets).
- Root-level legacy `utils/` package (removed after decoupling data loading from that path hack).
- Vendored `flow_matching/` source tree and its local tests/package setup files (migrated to external dependency mode).

## Data Policy

- Repository ships **no raw trajectory data** and **no checkpoints**.
- Users must place prepared public DiDi datasets under `./data/` (see `data/README.md`).

## Reproducibility Note

- This package is scoped for public training/inference/evaluation code.
- Internal benchmark orchestration scripts and private-data workflows are intentionally excluded to keep release boundaries clear.
