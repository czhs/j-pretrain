#!/bin/bash
# Reproducible env setup for j-pretrain (RTX 4090, driver 535 / CUDA 12.2).
# Creates conda env `jpre` with a pinned ML stack (cu121 wheels).
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV=jpre
if ! conda env list | grep -qE "^${ENV}\s"; then
  conda create -y -n "$ENV" python=3.11
fi
conda activate "$ENV"

python -m pip install --upgrade pip
# PyTorch cu121 (compatible with driver >=525; 4090)
python -m pip install "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cu121
# Core ML / data stack (pinned)
python -m pip install \
  "transformers==4.46.3" \
  "datasets==3.1.0" \
  "tokenizers==0.20.3" \
  "safetensors==0.4.5" \
  "accelerate==1.1.1" \
  "huggingface_hub==0.26.2" \
  "numpy==2.1.3" \
  "pandas==2.2.3" \
  "scipy==1.14.1" \
  "matplotlib==3.9.2" \
  "pyyaml==6.0.2" \
  "pyarrow==18.0.0" \
  "tqdm==4.67.1" \
  "pytest==8.3.3"

echo "=== versions ==="
python - <<'PY'
import torch, transformers, datasets, tokenizers, safetensors, numpy, scipy, pandas, matplotlib
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(), "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("tokenizers", tokenizers.__version__)
print("numpy", numpy.__version__, "scipy", scipy.__version__)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
echo "SETUP_ENV_DONE"
