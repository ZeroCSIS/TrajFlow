# [TrajFlow: nationwide Pseudo GPS Trajectory Generation with Flow Matching Models] Implementation

## Setup
```bash
conda env create -f environment.yml
conda activate flow_matching_py311
pip install -r requirements.txt
```

Notes:
- `flow_matching` is used as an external dependency (installed from the official GitHub repo via `requirements.txt`).
- This repository no longer vendors a local `flow_matching/` copy.

## Data
This repository does not ship raw trajectories, private data, or model weights.
Place prepared public DiDi data under `./data/` with folders like:
- `./data/DiDiTaxi_Chengdu`
- `./data/DiDiTaxi_XiAn`

## Training
```bash
python train.py --config ./src/config/config_chengdu.yaml
# XiAn:
# python train.py --config ./src/config/config_xian.yaml
```

## Inference
```bash
python generate.py \
  --config ./outputs/run_YYYYMMDD_HHMMSS/config.yaml \
  --checkpoint ./outputs/models/run_YYYYMMDD_HHMMSS/best_model.pt
```

## Evaluation
```bash
python eval_simple.py --result_dir /path/to/generation_folder
```
