# Mac Setup

This repo can use the same conda environment name as Windows: `conn_gui`.
On macOS, do not install `pytorch-cuda`; install the normal macOS PyTorch
packages instead.

## 1. Clone the repo

```bash
git clone https://github.com/JacquiLJQ/info-extraction-pipeline.git
cd info-extraction-pipeline
```

## 2. Create the conda environment

```bash
conda env create -f environment-macos.yml
conda activate conn_gui
```

If the environment already exists:

```bash
conda activate conn_gui
conda env update -f environment-macos.yml --prune
```

## 3. Restore local-only model/data files

The large zip files are intentionally ignored by git. Copy these files from
the Windows machine or another local backup into the repo root:

```text
AllData.zip
pretrainedWeights.zip
```

Then unzip them:

```bash
unzip AllData.zip
unzip pretrainedWeights.zip
```

The pipeline expects these paths:

```text
pretrainedWeights/componentDetection/best.pt
pretrainedWeights/junctionJumpDetection/best.pt
pretrainedWeights/orientationDetection/best_model.pt
pretrainedWeights/hawp/bestv3.pth
```

## 4. Verify the install

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("mps built:", torch.backends.mps.is_built())
print("mps available:", torch.backends.mps.is_available())
PY
```

## 5. Run

For the GUI:

```bash
python gui_red_flag_fixer.py
```

For the full pipeline:

```bash
python run_pipeline.py --config pipeline_config.yaml
```

The default config uses CPU because it is the most compatible option on macOS.
To try Apple Silicon GPU acceleration, run with MPS explicitly:

```bash
python run_pipeline.py --config pipeline_config.yaml --device mps
```

If MPS gives an operator/device error, return to the default CPU path.
