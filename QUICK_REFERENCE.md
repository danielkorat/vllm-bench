# vLLM Experiment Automation — Quick Reference

## Prerequisites

Enter the container first (all scripts run **inside** the container):

```bash
docker exec -it vllm-test bash
cd /root/vllm-bench
./experiment_utils.py check   # verify vLLM is available
```

---

## Running Experiments

### Sanity test (recommended before a full run)
```bash
./run_experiments.py --sanity
```
Runs 1 model, tp=2, 8-token I/O, 4 prompts. Takes ~5 minutes.

### Full benchmark suite
```bash
./run_experiments.py
```
Tests all combinations: 3 models × 3 TP values × 2 quant modes × 2 eager modes
(fp8+eager=false is skipped automatically).
Results saved to `./experiment_results/<timestamp>/`.

### Background run (with auto-resume)
```bash
./run_background.sh
```
Detaches from the terminal. Automatically skips already-completed experiments on
restart (`--resume` is passed by default). Use `--no-resume` to force a fresh run.
```bash
tail -f bg_run_*.log              # follow live output
kill $(cat experiment_bg.pid)     # stop the run
```

### Subset / custom parameters
```bash
# Single model
./run_experiments.py --models openai/gpt-oss-20b

# Specific TP + quantization
./run_experiments.py --models openai/gpt-oss-20b --tp 4 8 --quantization none fp8

# Override benchmark params
./run_experiments.py --input-len 512 --output-len 512 --concurrency 16 --num-prompts 80

# Override max model context length
./run_experiments.py --max-model-len 4096

# Resume a partial run
./run_experiments.py --resume

# Override sanity defaults
./run_experiments.py --sanity --tp 4 --input-len 16 --num-prompts 8

# All options
./run_experiments.py --help
```

---

## Monitoring

```bash
./experiment_utils.py status        # current state
./experiment_utils.py logs          # tail latest logs
watch -n 1 'xpu-smi stats -d 3'    # GPU usage (separate terminal)
```

---

## Analyzing Results

```bash
./analyze_results.py                              # latest results
./analyze_results.py --results-dir ./path/to/run  # specific run

cat experiment_results/<timestamp>/summary.txt
cat experiment_results/<timestamp>/results_summary.csv
```

---

## Utilities

```bash
./experiment_utils.py stop     # kill vLLM server + bench processes
./experiment_utils.py backup   # backup results before new run
./experiment_utils.py clean    # remove results (prompts confirmation)
./experiment_utils.py check    # verify prerequisites
./experiment_utils.py --help   # all commands
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `vllm: command not found` | You're outside the container — run `docker exec -it vllm-test bash` first |
| Server won't start | `./experiment_utils.py stop`, then retry |
| OOM / crashes | Lower `--gpu-memory-util` (default 0.9) or reduce `--max-model-len` |
| Disk full | `./experiment_utils.py backup && ./experiment_utils.py clean` |
| Container not running | `docker start vllm-test` |

---

## Files

| File | Purpose |
|---|---|
| `run_experiments.py` | Main runner — full suite + `--sanity` + `--resume` mode |
| `run_background.sh` | Background launcher with auto-resume and PID tracking |
| `experiment_utils.py` | Utility commands (status, stop, backup, …) |
| `analyze_results.py` | Post-run analysis and CSV/text reports |
| `experiment_common.py` | Shared dataclasses and logging |
| `EXPERIMENT_AUTOMATION.md` | Full documentation |

---

## Default Parameters

| Parameter | Full run | Sanity |
|---|---|---|
| Models | gpt-oss-20b, Qwen3-30B, Qwen3-4B | Qwen3-4B only |
| Tensor Parallelism | 2, 4, 8 | 2 |
| Quantization | none, fp8 | none, fp8 |
| Enforce Eager | true, false | true, false |
| Input / Output len | 1024 / 1024 | 8 / 8 |
| Max model len | 10000 | 2048 |
| Concurrency | 32 | 2 |
| Num prompts | 160 | 4 |
| Startup timeout | 300 s | 180 s |
| Benchmark timeout | 1800 s | 300 s |
