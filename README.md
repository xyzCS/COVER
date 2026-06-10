# COVER

COVER is the open-source code for [Stop the Flip-Flop: Context-Preserving Verification for Fast Revocable Diffusion Decoding](https://www.semanticscholar.org/paper/Stop-the-Flip-Flop%3A-Context-Preserving-Verification-Xiang-Wei/ab8612a612d7022d7a3b38c1fe2d37cf101f0305), accepted to ICML 2026. It provides a reproduction package for Cache Override Verification on diffusion language models. The repository contains three evaluation targets:

- `dream_instruct/`: Dream-v0-Instruct-7B with COVER.
- `llada_base/`: LLaDA-8B-Base with COVER.
- `llada_ins/`: LLaDA-8B-Instruct / LLaDA-1.5 with COVER, plus WINO and Saber baselines.

The package intentionally excludes generated `results/`, logs, and bytecode caches.

## Environment

A single Python environment can be used for all three subprojects.

```bash
cd /path/to/COVER_OS
conda create -n cover python=3.12 -y
conda activate cover
pip install -r requirement.txt
```

For HumanEval/MBPP code execution metrics, enable unsafe code evaluation:

```bash
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
```

Most scripts set these variables internally, but exporting them is useful for direct CLI runs.

## Directory Layout

```text
COVER/
  dream_instruct/
    code/                 # Dream lm_eval fork, COVER model/generation, metrics
    scripts/              # run_cover_256.sh, run_cover_512.sh
  llada_base/
    code/                 # LLaDA-base lm_eval fork, COVER generator/model
    scripts/              # run_cover_256.sh, run_cover_512.sh
  llada_ins/
    code/                 # LLaDA-instruct dllm_eval fork
    scripts/              # COVER, WINO, Saber scripts for 256/512
  requirement.txt
```

## Running COVER

All scripts should be launched from the corresponding `code/` directory. All reported experiments were run on four NVIDIA H200 GPUs; the example commands therefore use `CUDA_VISIBLE_DEVICES=0,1,2,3`.

### Dream-Instruct COVER

```bash
cd /path/to/COVER_OS/dream_instruct/code
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ../scripts/run_cover_256.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ../scripts/run_cover_512.sh
```

The Dream scripts use the selected balanced hyperparameters for the four tasks: HumanEval, MBPP, GSM8K, and MATH500. Use `ONLY_TASK` to run one task:

```bash
ONLY_TASK=math500 CUDA_VISIBLE_DEVICES=0,1,2,3 bash ../scripts/run_cover_256.sh
```

### LLaDA-Base COVER

```bash
cd /path/to/COVER_OS/llada_base/code
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ../scripts/run_cover_256.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ../scripts/run_cover_512.sh
```

The LLaDA-base scripts use paper-tuned per-task hyperparameters and few-shot settings.

### LLaDA-Instruct COVER

```bash
cd /path/to/COVER_OS/llada_ins/code
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ../scripts/run_cover_256.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ../scripts/run_cover_512.sh
```

By default these use `GSAI-ML/LLaDA-8B-Instruct`. LLaDA-1.5 wrappers are also provided:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ../scripts/run_cover_llada15_256.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ../scripts/run_cover_llada15_512.sh
```

Equivalently, set `MODEL_PATH=GSAI-ML/LLaDA-1.5` when calling `run_cover_256.sh` or `run_cover_512.sh`.

## LLaDA-Instruct Baselines

`llada_ins` includes two revocable decoding baselines: WINO and Saber.

### WINO

```bash
cd /path/to/COVER_OS/llada_ins/code
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ../scripts/run_wino_256.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ../scripts/run_wino_512.sh
```

### Saber

```bash
cd /path/to/COVER_OS/llada_ins/code
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ../scripts/run_saber_256.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ../scripts/run_saber_512.sh
```

Run a single task with `ONLY_TASK`:

```bash
ONLY_TASK=humaneval CUDA_VISIBLE_DEVICES=0,1,2,3 bash ../scripts/run_wino_512.sh
ONLY_TASK=math500 CUDA_VISIBLE_DEVICES=0,1,2,3 bash ../scripts/run_saber_512.sh
```

## Flip-Flop Revision Ratio

All subprojects include a `flip_flop_ratio.py` utility that computes the paper-style effective revision ratio from `samples_*.jsonl`.

Paper definitions:

- COVER: `total = ReMask + Replace`, `effective = ChangedAfterReMask + Replace`.
- WINO/Saber: `total = ReMask`, `effective = ReMask - FlipFlop`.
- `ratio = effective / total`.

Locations:

```text
dream_instruct/code/metrics/flip_flop_ratio.py
llada_base/code/flip_flop_ratio.py
llada_ins/code/metrics/flip_flop_ratio.py
```

The LLaDA-Instruct WINO/Saber scripts automatically compute and print the ratio after each task, saving:

```text
<output_path>/flip_flop_ratio.log
<output_path>/flip_flop_ratio.txt
```

## Citation

If you use this code, please cite:

```bibtex
@article{Xiang2026StopTF,
  title={Stop the Flip-Flop: Context-Preserving Verification for Fast Revocable Diffusion Decoding},
  author={Yanzheng Xiang and Lan Wei and Yizhen Yao and Qinglin Zhu and Hanqi Yan and Chen Jin and Philip Alexander Teare and Dandan Zhang and Lin Gui and Amrutha Saseendran and Yulan He},
  journal={ArXiv},
  year={2026},
  volume={abs/2602.06161},
  url={https://api.semanticscholar.org/CorpusID:285401255}
}
```
