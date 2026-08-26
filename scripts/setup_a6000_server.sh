#!/bin/bash
# Sets up DietAI24 to run fully locally on a new server, reproducing what was built/debugged
# in the original session against a 2x RTX 4090 box. Run each section manually (not blindly as
# one script) since the CUDA-version-matching step depends on this machine's actual driver.
set -e

# ============================================================
# 0. Prerequisites: clone the repo (if not already done), check GPUs/driver
# ============================================================
# git clone https://github.com/yongshenglian/dietai26.git DietAI24
# cd DietAI24

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo "Look at the 'CUDA Version' line in plain 'nvidia-smi' output below -- that's this"
echo "driver's ceiling. Pick a CUDA_TOOLKIT_VERSION at or below it (we used 12.8 on a driver"
echo "capped at 12.8; if this server's driver supports 12.9+ you can likely just use pip's"
echo "prebuilt vllm wheel directly instead of the from-source build in section 2)."
nvidia-smi | grep "CUDA Version"

# ============================================================
# 1. Python env for the DietAI24 client code (langchain, etc.)
# ============================================================
conda create -n dietai24 python=3.12 -y
source ~/miniconda3/etc/profile.d/conda.sh   # adjust path if conda lives elsewhere
conda activate dietai24
pip install \
  langchain==1.3.1 \
  langchain-openai==1.2.1 \
  langchain-community==0.4.1 \
  langchain-chroma==1.1.0 \
  langchain-classic==1.0.7 \
  langchain-core==1.4.0 \
  langchain-text-splitters==1.1.2 \
  chromadb==1.5.9 \
  openai==2.37.0 \
  pandas==3.0.3 \
  openpyxl==3.1.5 \
  python-dotenv==1.2.2

# ============================================================
# 2. vLLM serving env -- build from source against a CUDA toolkit matched to this
# ============================================================
# WHY from source: pip's prebuilt vllm wheel is only offered for CUDA 12.9/13.0. If this
# server's driver caps below that (check step 0), the prebuilt wheel fails at CUDA init with
# "NVIDIA driver is too old". Building from source against a self-contained, driver-matched
# CUDA toolkit sidesteps that -- and doesn't touch any system packages (fully sudo-free).
CUDA_TOOLKIT_VERSION=12.8.1   # <-- change to match what you found in step 0

conda create -n vllm-serve python=3.12 -y
conda activate vllm-serve
conda install -y -c "nvidia/label/cuda-${CUDA_TOOLKIT_VERSION}" -c conda-forge \
  cuda-toolkit gxx_linux-64=11 gcc_linux-64=11

# CUDA 12.4 specifically hits a known issue: nvcc silently ignores -std=c++20 and falls back to
# an older standard, breaking the build with cryptic "namespace std has no member ..." errors.
# 12.6+ doesn't have this problem. If you must use 12.4, expect that failure and bump the
# version instead of debugging the symptom.

# vLLM's build expects the classic NVIDIA-installer layout (lib64/stubs/...); conda's package
# uses lib/ instead. This symlink fixes FlashInfer's JIT kernel builds (and anything else that
# assumes lib64) without needing sudo or touching the system.
ln -s lib "$CONDA_PREFIX/lib64"

git clone --depth 1 https://github.com/vllm-project/vllm.git ~/models/vllm-src
cd ~/models/vllm-src
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=16          # tune to available cores
export TORCH_CUDA_ARCH_LIST="8.6"   # A6000 = Ampere, compute capability 8.6 (RTX 4090 was 8.9 -- different!)
uv pip install -e . --torch-backend=auto
# This compiles CUDA kernels from scratch -- expect 20-40+ minutes depending on core count.

# ============================================================
# 3. Serve the models
# ============================================================
# With 4x A6000 (48GB each = 192GB total) you have much more headroom than the 2x24GB box this
# was originally built on. Worth reconsidering quantization/quality tradeoffs here rather than
# copying the AWQ-INT4 choice blindly -- that was specifically chosen to fit a 24GB card. On a
# 48GB card you could likely run FP8 (Ampere doesn't have native FP8 tensor cores like Ada/Hopper
# do, so this needs testing) or a larger/less-quantized checkpoint. Below reproduces the known-
# working setup as a starting point; not necessarily the best use of 4x48GB.

export HF_HOME=~/models   # wherever you want weights cached

# GPU 0: generate (chat + vision)
CUDA_VISIBLE_DEVICES=0 vllm serve cyankiwi/Qwen3.8-27B-AWQ-INT4 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.95 \
  --enforce-eager \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"reasoning_effort":"low"}' \
  --port 8000 &

# GPU 1: embeddings (dedicated embedding model -- pooling from the 27B chat model measured
# noticeably worse retrieval quality, see analyze_image.py comments)
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-Embedding-0.6B \
  --tensor-parallel-size 1 \
  --runner pooling \
  --convert embed \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager \
  --port 8001 &

# GPUs 2-3 are free. Options: --tensor-parallel-size 2 across GPU0+2 (or similar) for higher
# throughput/quality on the generate model, or a second independent model. Worth deciding
# deliberately rather than defaulting to the 2-GPU-box layout.

# ============================================================
# 4. Data the repo doesn't ship (all gitignored on purpose)
# ============================================================
# - code/.env with OPENAI_API_KEY (only needed if you want any OpenAI-backed paths; the local
#   pipeline built in analyze_image.py doesn't need it)
# - FNDDS/ and NHANES/ reference data
# - ASA24 / Nutrition5k raw images if you want to rerun the evaluation harness
# - chroma_fndds_db/ vector index: will rebuild automatically on first analyze_image.py run
#   (~5,600 embedding calls, a few minutes), or copy it over from the original box to skip that
