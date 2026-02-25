# vLLM Experiment Automation

Automated benchmarking system for running comprehensive vLLM experiments across multiple model configurations.

## Overview

This automation suite runs up to **30 different configurations** (fp8+eager=false is skipped automatically) combining:
- **Models**: 3 models (openai/gpt-oss-20b, Qwen/Qwen3-30B-A3B, Qwen/Qwen3-4B-Thinking-2507)
- **Tensor Parallelism**: 3 values (2, 4, 8)
- **Quantization**: 2 options (none, fp8)
- **Enforce Eager**: 2 options (true, false)

All scripts run **inside** the Docker container – no `docker exec` wrappers are used.

## Prerequisites

1. Docker container must be running:
   ```bash
   # If not already running, start it:
   docker run -t -d --shm-size 10g --net=host --ipc=host --privileged \
     -e http_proxy=http://proxy-dmz.intel.com:912 \
     -e https_proxy=http://proxy-dmz.intel.com:912 \
     -e HTTP_PROXY=http://proxy-dmz.intel.com:912 \
     -e HTTPS_PROXY=http://proxy-dmz.intel.com:912 \
     -e no_proxy=localhost,127.0.0.1,0.0.0.0 \
     -e NO_PROXY=localhost,127.0.0.1,0.0.0.0 \
     -e HF_TOKEN=${HF_READ_TOKEN} \
     --name=vllm-test \
     --device /dev/dri:/dev/dri \
     -v ~/.cache/huggingface:/root/.cache/huggingface \
     -v /dev/dri/by-path:/dev/dri/by-path \
     -v /root/dkorat/vllm-bench/:/root/vllm-bench \
     --entrypoint= intel/vllm:0.14.1-xpu /bin/bash
   ```

2. Enter the container:
   ```bash
   docker exec -it vllm-test bash
   cd /root/vllm-bench
   ```

3. (Optional) Verify vLLM is available:
   ```bash
   ./experiment_utils.py check
   ```

## Quick Start

### Sanity Test (Recommended First)

```bash
./run_experiments.py --sanity
```

Runs a minimal check (1 model, tp=2, 8-token I/O, 4 prompts) to verify the full
pipeline works before committing to a multi-hour full run.

### Run All Experiments

```bash
./run_experiments.py
```

This will:
- Run up to 30 configuration combinations (fp8+eager=false skipped automatically)
- Handle errors gracefully (skip failed configs)
- Save results to `./experiment_results/<timestamp>/`
- Generate a summary report
- Create detailed logs for each run

**Expected Duration**: Several hours (depends on model sizes and hardware)

### Analyze Results

After experiments complete:

```bash
./analyze_results.py
```

Or specify a custom results directory:

```bash
./analyze_results.py --results-dir ./experiment_results
```

This generates:
- Console output with statistics and comparisons
- `detailed_analysis.txt` - Full details of all experiments
- `results_summary.csv` - Spreadsheet-friendly format

## Output Structure

```
experiment_results/
└── YYYYMMDD_HHMM/                                            # Timestamped run dir
    ├── summary.txt                                           # Overall summary
    ├── detailed_analysis.txt                                 # Detailed analysis
    ├── results_summary.csv                                   # CSV export
    ├── openai_gpt-oss-20b_tp2_quant-none_eager-true_results.json
    ├── openai_gpt-oss-20b_tp2_quant-none_eager-false_results.json
    ├── ... (all result files)
    └── logs/
        ├── openai_gpt-oss-20b_tp2_quant-none_eager-true_server.log
        ├── openai_gpt-oss-20b_tp2_quant-none_eager-true_benchmark.log
        └── ... (all log files)
```

## Configuration

Customize experiments using command-line options:

```bash
# Sanity test with default settings
./run_experiments.py --sanity

# Sanity test overriding specific params
./run_experiments.py --sanity --tp 4 --input-len 16 --num-prompts 8

# Full run with specific models
./run_experiments.py --models openai/gpt-oss-20b Qwen/Qwen3-30B-A3B

# Test specific tensor parallelism values
./run_experiments.py --tp 4 8

# Test specific quantization modes (use 'none' for no quantization)
./run_experiments.py --quantization none fp8

# Override benchmark parameters directly
./run_experiments.py --input-len 512 --output-len 512 --concurrency 16 --num-prompts 80

# Adjust timeouts
./run_experiments.py --timeout-startup 600 --timeout-benchmark 3600

# See all options
./run_experiments.py --help
```

Default values (full run):

```python
models          = ['openai/gpt-oss-20b', 'Qwen/Qwen3-30B-A3B', 'Qwen/Qwen3-4B-Thinking-2507']
tp              = [2, 4, 8]
quantization    = ['none', 'fp8']   # 'none' omits --quantization flag
enforce_eager   = ['true', 'false']
timeout_startup = 300               # 5 minutes
timeout_benchmark = 1800            # 30 minutes
input_len       = 1024
output_len      = 1024
concurrency     = 32
num_prompts     = 160
```

Default values (sanity test via `--sanity`):

```python
models          = ['Qwen/Qwen3-4B-Thinking-2507']
tp              = [2]
quantization    = ['none', 'fp8']
enforce_eager   = ['true', 'false']
timeout_startup = 180
timeout_benchmark = 300
input_len       = 8
output_len      = 8
concurrency     = 2
num_prompts     = 4
```

## Features

### Error Handling
- Gracefully skips failed configurations
- Continues with remaining experiments
- Logs all errors for debugging
- Provides detailed failure reasons in summary

### Progress Tracking
- Real-time colored console output
- Timestamps for all operations
- Progress indicators during startup
- Health checks before benchmarking

### Result Organization
- Descriptive filenames with all parameters
- Separate server and benchmark logs
- JSON results for programmatic analysis
- CSV export for spreadsheets

### Summary Report
- Total experiments run
- Success/failure counts
- Success rate percentage
- List of all successful configs
- List of all failed configs with reasons
- Total duration

## Monitoring

### Check Current Status
```bash
./experiment_utils.py status
```

### Watch Live Logs (from inside the container)
```bash
./experiment_utils.py logs
# or directly:
tail -f experiment_results/*/logs/*_server.log
```

### Monitor GPU Usage (separate terminal)
```bash
watch -n 1 'xpu-smi stats -d 3'
```

## Troubleshooting

### vLLM Not Found in PATH
You are running outside the container. Enter it first:
```bash
docker exec -it vllm-test bash
cd /root/vllm-bench
```

### Stuck Experiments
```bash
# Kill hanging vLLM processes (run inside the container)
./experiment_utils.py stop

# Restart the experiment script
./run_experiments.py
```

### Container Not Running
```bash
docker ps -a | grep vllm-test
docker start vllm-test  # If stopped
```

### Disk Space Issues
```bash
# Check available space
df -h

# Clean up old results if needed
rm -rf experiment_results.old/
mv experiment_results/ experiment_results.old/
```

### Memory Issues
If experiments fail due to OOM, use command-line options or edit the script to reduce:
- `--gpu-memory-util 0.9` → reduce to 0.8 or 0.7
- `--max-model-len 16384` → reduce to 8192 or 4096
- `--max-num-batched-tokens 8192` → reduce to 4096

## Advanced Usage

### Run Subset of Experiments

Use command-line options to test specific configurations:

```bash
# Test only one model
./run_experiments.py --models openai/gpt-oss-20b

# Test only specific TP values
./run_experiments.py --tp 4 8

# Test only without quantization
./run_experiments.py --quantization none

# Combine multiple filters
./run_experiments.py --models openai/gpt-oss-20b \
                     --tp 4 \
                     --quantization none fp8
```

### Resume After Failure

The script automatically skips completed experiments if you've organized results by timestamp:

```bash
# Create timestamped backup
cp -r experiment_results experiment_results_$(date +%Y%m%d_%H%M%S)

# Modify script to skip completed configs
# (manual edit required based on summary.txt)
```

### Custom Benchmark Parameters

Use CLI flags (no code edits needed):

```bash
./run_experiments.py \
    --input-len 1024 \
    --output-len 1024 \
    --concurrency 32 \
    --num-prompts 160
```

## Performance Tips

1. **Run during low-usage periods** - Ensures consistent benchmarking
2. **Monitor temperature** - Thermal throttling affects results
3. **Run multiple times** - Average results for reliability
4. **Consistent environment** - Same models, system state, etc.

## Example Output

```
========================================
Starting experiment: openai_gpt-oss-20b_tp4_quant-off_eager-true
Model: openai/gpt-oss-20b
Tensor Parallelism: 4
Quantization: off
Enforce Eager: true
========================================
[2026-02-24 10:30:15] Starting vLLM server...
[2026-02-24 10:30:20] Waiting for vLLM server to be ready (timeout: 300s)...
[2026-02-24 10:31:45] ✓ Server is ready!
[2026-02-24 10:31:45] Running benchmark...
[2026-02-24 10:45:12] ✓ Experiment completed successfully: openai_gpt-oss-20b_tp4_quant-off_eager-true
[2026-02-24 10:45:12] Throughput: 2543.67, Mean Latency: 125.34
```

## Summary Report Example

```
vLLM Benchmark Experiment Summary
==================================
Generated: 2026-02-24 18:30:45

Total Experiments: 36
Successful: 34
Failed: 2
Success Rate: 94.44%

Total Duration: 8h 15m 32s

Successful Experiments:
----------------------
  ✓ openai_gpt-oss-20b_tp2_quant-off_eager-true
  ✓ openai_gpt-oss-20b_tp2_quant-off_eager-false
  ... (32 more)

Failed Experiments:
------------------
  ✗ Qwen_Qwen3-30B-A3B_tp8_quant-fp8_eager-true (server startup failed)
  ✗ Qwen_Qwen3-4B-Thinking-2507_tp2_quant-fp8_eager-false (benchmark timeout)

Results Location: ./experiment_results
Logs Location: ./experiment_results/logs
```

## Questions?

Check logs in `experiment_results/<timestamp>/logs/` for detailed error messages and debugging information.

Reference: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

