# [ICML 2026] PhoStream: Benchmarking Real-World Streaming for Omnimodal Assistants in Mobile Scenarios

## Setup

### 0. Prerequisites

- **ffmpeg** (includes `ffprobe`). Install via your system package manager:

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Conda
conda install ffmpeg
```

### 1. Download Videos

Download the four tar.gz archives from [PhoStream on HuggingFace](https://huggingface.co/datasets/lucky-lance/PhoStream) into the `videos/` directory:

```text
videos/
  EgoBlind_reencoded_2fps.tar.gz
  record_file_reencoded_2fps.tar.gz
  phone_class_reencoded_2fps.tar.gz
  youtube_dl_reencoded_2fps.tar.gz
```

### 2. Extract Videos

```bash
cd videos
tar -xf EgoBlind_reencoded_2fps.tar.gz
tar -xf record_file_reencoded_2fps.tar.gz
tar -xf phone_class_reencoded_2fps.tar.gz
tar -xf youtube_dl_reencoded_2fps.tar.gz
cd ..
```

### 3. Create Symlinks

```bash
python setup_symlinks.py
```

### 4. Install Dependencies

```bash
conda create -n phostream python=3.12 -y
conda activate phostream
pip install -r requirements.txt
```

> **Flash Attention** is required for the HF backend (Qwen evaluation). Download the appropriate wheel for your CUDA/PyTorch version from [flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases/). For CUDA 12.8 + PyTorch 2.7 + Python 3.12:
>
> ```bash
> pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
> ```
>

## Evaluation

### Qwen (HF backend)

```bash
# Start 10-GPU servers
MODEL_PATH=/path/to/qwen3-vl-8b \
bash scripts_parallel/start_hf_servers.sh

# Inference (10 GPUs) + scoring (32 concurrent)
STREAMEVAL_JUDGER_API_BASE=https://your-judge-api.com/v1 \
STREAMEVAL_JUDGER_API_KEY=your-judge-api-key \
STREAMEVAL_JUDGER_MODEL=qwen3-235b-a22b-instruct-2507 \
NUM_WORKERS=10 \
NUM_SCORERS=32 \
bash scripts_parallel/run_eval.sh hf qwen3_vl

# Stop servers
bash scripts_parallel/start_hf_servers.sh stop
```

### Gemini (cloud API)

```bash
STREAMEVAL_GEMINI_3_API_BASE=your-gemini-api-host.com \
STREAMEVAL_GEMINI_3_API_KEY=your-gemini-api-key \
STREAMEVAL_JUDGER_API_BASE=https://your-judge-api.com/v1 \
STREAMEVAL_JUDGER_API_KEY=your-judge-api-key \
STREAMEVAL_JUDGER_MODEL=qwen3-235b-a22b-instruct-2507 \
NUM_WORKERS=64 \
NUM_SCORERS=32 \
bash scripts_parallel/run_eval.sh gemini gemini_3_pro
```

### Score-only mode (skip inference)

```bash
bash scripts_parallel/run_eval.sh score qwen3_vl /path/to/output.jsonl
```

## Citation

```bibtex
@article{lu2026phostream,
  title={PhoStream: Benchmarking Real-World Streaming for Omnimodal Assistants in Mobile Scenarios},
  author={Lu, Xudong and Guan, Huankang and Bo, Yang and Chen, Jinpeng and Guo, Xintong and Li, Shuhan and Liu, Fang and Sun, Peiwen and Li, Xueying and Zhang, Wei and others},
  journal={arXiv preprint arXiv:2601.22575},
  year={2026}
}
```
