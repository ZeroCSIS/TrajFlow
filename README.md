# TrajFlow: nationwide Pseudo GPS Trajectory Generation with Flow Matching Models

Official implementation of **TrajFlow**.

![TrajFlow Motivation](assets/Fig1.png)

## Paper
- OpenReview page: https://openreview.net/forum?id=BDOldEjwCE
- PDF: https://openreview.net/pdf?id=BDOldEjwCE

TrajFlow is a flow-matching based framework for pseudo GPS trajectory generation.
It targets multi-scale trajectory generation and supports training/inference on public-style trajectory datasets.

## Open-Source Scope
- This repository provides the training/inference/evaluation pipeline for TrajFlow.
- The main paper conclusions are validated on **BW** data, which is commercial/private and not open-sourced here.
- Open-source support is provided for **DiDi Chengdu/XiAn style data** to verify that the pipeline works on public-style data.

## Setup
```bash
conda env create -f environment.yml
conda activate flow_matching_py311
pip install -r requirements.txt
```

Notes:
- `flow_matching` is installed as an external dependency via `requirements.txt`.
- This repository does not vendor a local `flow_matching/` copy.

## Data Policy
- This repository does **not** ship raw trajectories, private data, or model checkpoints.
- We do **not** redistribute DiDi datasets in this repository.
- Please obtain DiDi data from official/authorized channels under your own compliance responsibility.

Expected local layout for testing:
- `./data/DiDiTaxi_Chengdu_traj`
- `./data/DiDiTaxi_XiAn_traj`

Required files for each dataset folder are listed in `data/README.md`.

## Usage

Training:
```bash
python train.py --config ./src/config/config_chengdu.yaml
# XiAn:
# python train.py --config ./src/config/config_xian.yaml
```

Generation:
```bash
python generate.py \
  --config ./outputs/run_YYYYMMDD_HHMMSS/config.yaml \
  --checkpoint ./outputs/models/run_YYYYMMDD_HHMMSS/best_model.pt
```

Evaluation:
```bash
python eval_simple.py --result_dir /path/to/generation_folder
```

## License
This codebase is released under **CC BY-NC 4.0** (`LICENSE`).

## Citation
If you use this repository, please cite:

```bibtex
@inproceedings{li2026trajflow,
  title={TrajFlow: nationwide Pseudo GPS Trajectory Generation with Flow Matching Models},
  author={Li, Peiran and Wang, Jiawei and Zhang, Haoran and Shi, Xiaodan and Koshizuka, Noboru and Shimizu, Chihiro and Jiang, Renhe},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026},
  url={https://openreview.net/forum?id=BDOldEjwCE}
}
```
