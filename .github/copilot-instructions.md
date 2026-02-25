# vLLM Benchmarking Automation - Repository Instructions

## Repository Purpose

Automated benchmarking system for Intel XPU vLLM inference across multiple model configurations. This repo systematically tests combinations of models, tensor parallelism, quantization modes, and execution strategies to identify optimal performance configurations.

## Goals

1. **Comprehensive Testing**: Run all combinations of vLLM configurations automatically
2. **Error Resilience**: Handle failures gracefully and continue with remaining experiments
3. **Data Collection**: Capture detailed performance metrics and logs for analysis
4. **Result Analysis**: Generate actionable insights from benchmark data

## Repository Structure

```
/root/vllm-bench/
├── run_experiments.py          # Main automation engine – runs ALL experiments
│                               #   --sanity flag activates sanity-test mode
│                               #   --resume skips already-completed experiments
├── run_background.sh           # Detached background launcher (auto-resume, PID file)
├── experiment_common.py        # Shared dataclasses (ExperimentConfig, ExperimentResult, Logger)
├── experiment_utils.py         # Utility commands (status, stop, clean, backup, check, logs)
├── analyze_results.py          # Results analysis and visualization
├── EXPERIMENT_AUTOMATION.md    # Comprehensive documentation
├── QUICK_REFERENCE.md          # Quick command reference (Markdown)
├── reference.sh                # Original manual benchmark reference commands
├── ai-code-slop/               # Web UI components (separate project)
├── experiment_results/         # Created at runtime (full experiments)
│   └── YYYYMMDD_HHMM/          # Timestamped run directory
│       ├── *_results.json      # Individual experiment results
│       ├── summary.txt         # Run summary
│       ├── detailed_analysis.txt  # Formatted human-readable analysis report
│       ├── raw_results.json    # Raw JSON dump of all result data
│       ├── results_summary.csv
│       └── logs/
└── sanity_test_results/        # Created at runtime (sanity tests, same structure)
```

## Technology Stack

- **Language**: Python 3 (migrated from bash for better maintainability)
- **Container**: Docker (intel/vllm:0.14.1-xpu)
- **Hardware**: Intel XPU (discrete GPUs accessed via /dev/dri)
- **LLM Framework**: vLLM with Intel XPU backend
- **Environment**: Intel corporate network with proxy configuration

## Critical Rules & Corrections

### 1. Quantization Flag Handling
**RULE**: Never use `--quantization off` or `--quantization none`
- ❌ WRONG: `--quantization off` (invalid flag value)
- ✅ CORRECT: Omit the `--quantization` flag entirely when not using quantization
- When quantization is disabled/none, simply don't add the flag to the command

```python
# Correct implementation
if config.quantization:
    cmd_parts.append(f"--quantization {config.quantization}")
# Otherwise, no --quantization flag is added
```

### 2. Language Preference
**RULE**: Use Python for all automation scripts
- Bash was replaced with Python for better error handling, type safety, and maintainability
- All new scripts must be Python 3
- Use argparse for CLI interfaces
- Use dataclasses for configuration objects

### 3. Execution Environment
**RULE**: `run_experiments.py` runs **inside** the container.
- Scripts are executed by calling them directly within the container shell.
- `check_environment()` checks `which vllm` (not docker container status).
- Never add `docker exec` wrappers back; they break oneAPI loading.
- To launch: `docker exec -it vllm-test bash`, then `./run_experiments.py` (or `./run_experiments.py --sanity` for a quick check).

### 4. Error Handling Strategy
**RULE**: Graceful degradation - skip failed experiments, continue with rest
- Log all errors with detailed context
- Track success/failure counts
- Generate summary at the end
- Never exit on first failure

### 5. File Organization
**RULE**: All experiment outputs must be organized with descriptive names
- Pattern: `{model}_{tp}{value}_quant-{mode}_eager-{bool}_results.json`
- Separate directories for results and logs
- Timestamp backups when preserving old results

### 6. Proxy Configuration
**RULE**: Intel proxy settings are mandatory for all Docker operations
- HTTP/HTTPS proxy: `http://proxy-dmz.intel.com:912`
- Must set both lowercase and uppercase env vars
- Add localhost to no_proxy exceptions

### 7. Script Executability
**RULE**: All .py scripts must be executable
- Run `chmod +x *.py` after creating new scripts
- Add shebang: `#!/usr/bin/env python3`

### 8. FP8 + Eager=False Exclusion
**RULE**: Always skip fp8 quantization with eager=false (known vLLM engine failure)
- This configuration fails with "RuntimeError: Engine core initialization failed"
- Scripts must filter out this combination before running experiments
- Log a warning when skipping: `Logger.warning(f"Skipping {model} tp={tp} quant=fp8 eager=false (known failure)")`

```python
# Correct implementation
for quant in quantization_options:
    for eager in enforce_eager_options:
        # Skip fp8 + eager=false (known to fail)
        if quant == 'fp8' and not eager:
            Logger.warning(f"Skipping fp8 eager=false (known failure)")
            continue
```

### 9. Proxy Bypass for Benchmarks
**RULE**: Set `no_proxy` env vars inside the benchmark and server shell scripts
- Intel proxy intercepts connections; must explicitly bypass local addresses.
- Export proxy bypass vars directly in the `/tmp/*.sh` scripts (not via docker exec `-e` flags).

```python
# Correct implementation
with open('/tmp/run_benchmark.sh', 'w') as f:
    f.write("\n".join([
        "#!/bin/bash",
        "source /opt/intel/oneapi/setvars.sh --force",
        "export no_proxy=localhost,127.0.0.1,0.0.0.0",
        "export NO_PROXY=localhost,127.0.0.1,0.0.0.0",
        benchmark_cmd,
    ]) + "\n")
```

### 10. oneAPI Environment Initialization
**RULE**: Both the server and benchmark scripts must be run with `bash --login` after explicitly sourcing `/opt/intel/oneapi/setvars.sh --force`.
- Scripts now run **inside** the container, so there is no `docker exec` wrapper.
- Write a `.sh` script to `/tmp/`, `os.chmod` it `0o755`, and launch with `bash --login /tmp/script.sh`.
- The `source /opt/intel/oneapi/setvars.sh --force` line inside the script loads the full XPU environment.
- ✅ CORRECT: write script locally → `bash --login /tmp/script.sh`
- ❌ WRONG: `bash -c "command"` without sourcing setvars; or using `docker exec` from outside the container.

```python
# Correct implementation
script = "/tmp/start_vllm_server.sh"
with open(script, 'w') as f:
    f.write("\n".join([
        "#!/bin/bash",
        "source /opt/intel/oneapi/setvars.sh --force",
        "export no_proxy=localhost,127.0.0.1,0.0.0.0",
        "export NO_PROXY=localhost,127.0.0.1,0.0.0.0",
        vllm_cmd,
    ]) + "\n")
os.chmod(script, 0o755)
subprocess.Popen(["bash", "--login", script], ...)
```

### 11. Health Check Implementation
**RULE**: Use curl -f and don't require non-empty response body
- The /health endpoint returns 200 OK with empty body (valid)
- Use `curl -f` to fail on HTTP errors (4xx, 5xx)
- Check only returncode == 0, NOT stdout content

```python
# Correct implementation
result = self.exec_local(f"curl -f -s http://localhost:{self.port}/health", timeout=10)
if result.returncode == 0:  # Don't check result.stdout.strip()
    Logger.success("Server is ready!")
    return True
```

### 12. Benchmark Result Validation
**RULE**: Use correct JSON field names from vllm bench output
- Use 'completed' for successful requests (not 'num_successes')
- Use 'failed' for failed requests (not 'num_failures')
- Use 'request_throughput' and 'mean_ttft_ms' for metrics
- Always validate benchmark actually succeeded (completed > 0, failed == 0)

```python
# Correct implementation
data = json.load(f)
failed_requests = data.get('failed', 0)
successful_requests = data.get('completed', 0)
if failed_requests > 0 or successful_requests == 0:
    raise Exception(f"Benchmark had {failed_requests} failures, {successful_requests} successes")
```

### 13. Server Startup Monitoring
**RULE**: Monitor vllm process health during startup, show logs on failure
- Check if process is running with `pgrep -f 'vllm serve'`
- Periodically verify process hasn't died during long startups
- Display last 50 lines of server logs when startup fails
- This helps debug initialization errors immediately

```python
# Correct implementation
def wait_for_server(self) -> bool:
    time.sleep(10)  # Let process initialize
    result = self.exec_local("pgrep -f 'vllm serve' | head -1", timeout=5)
    if result.returncode != 0 or not result.stdout.strip():
        Logger.error("vLLM process not running!")
        self._show_server_logs()
        return False
```

### 14. Benchmark Host Configuration
**RULE**: Use `0.0.0.0` as the --host for benchmark client connections (matches official.sh)
- ✅ CORRECT: `--host 0.0.0.0` (as used in official Intel reference script)
- Server binds to `0.0.0.0:8000`, benchmark client connects to same
- This is the validated working configuration from official.sh

```python
# Correct implementation (from official.sh)
def build_benchmark_command(self, config: ExperimentConfig) -> str:
    return f"""vllm bench serve \
        --host 0.0.0.0 \
        --port {self.port} \
        ..."""
```

### 15. Clean Up Before Experiments
**RULE**: Always kill previous vLLM servers AND benchmark processes before starting
- Kill both `vllm serve` and `vllm bench` processes
- Benchmark processes can hang and consume resources
- Use pkill -f to match full command line

```python
# Correct implementation
def stop_vllm_server(self):
    if self._server_process is not None:
        try:
            self._server_process.kill()
            self._server_process.wait(timeout=10)
        except Exception:
            pass
        self._server_process = None
    for pattern in ("'vllm serve'", "'vllm bench'"):
        try:
            self.exec_local(f"pkill -f {pattern}", capture_output=False)
        except subprocess.CalledProcessError:
            pass
    time.sleep(5)
```

### 16. Timestamped Result Directories
**RULE**: Create timestamped subdirectories for each experiment run in Israel Time
- Use `ZoneInfo("Asia/Jerusalem")` for timezone
- Format: `YYYYMMDD_HHMM` (e.g., `20260225_1430`)
- Each run creates a new subdirectory under base results directory
- Prevents result overwrites and provides chronological organization

```python
# Correct implementation
from zoneinfo import ZoneInfo
from datetime import datetime

israel_tz = ZoneInfo("Asia/Jerusalem")
timestamp = datetime.now(israel_tz).strftime("%Y%m%d_%H%M")
base_results_dir = Path(results_dir)
self.results_dir = base_results_dir / timestamp
self.results_dir.mkdir(parents=True, exist_ok=True)
```

### 17. analyze_results.py Output Files
**RULE**: `analyze_results.py` writes three output files per results directory:
- `detailed_analysis.txt` — full formatted human-readable report: overall stats, per-model table, best configs per model, and best configs overall
- `raw_results.json` — raw JSON array of `{filename, config, data}` for every result file
- `results_summary.csv` — one row per experiment, spreadsheet-friendly

The same formatted content is also printed to stdout (and captured in `bg_run_*.log`).

## Configuration Parameters

### Models to Test (Default)
```python
[
    'openai/gpt-oss-20b',
    'Qwen/Qwen3-30B-A3B',
    'Qwen/Qwen3-4B-Thinking-2507'
]
```

### Tensor Parallelism Values
```python
[2, 4, 8]
```

### Quantization Modes
```python
['none', 'fp8']  # 'none' means omit --quantization flag
```

### Eager Execution Modes
```python
[True, False]  # Always test both
```

### Default Timeouts
- Server startup: 300 seconds (5 minutes)
- Benchmark: 1800 seconds (30 minutes)

### Max Model Length
```python
FULL_BENCH_DEFAULTS: max_model_len = 10000
SANITY_DEFAULTS:     max_model_len = 2048
```
Exposed as `--max-model-len` CLI argument.

## Common Commands

### Run Sanity Test (Quick Validation)
```bash
./run_experiments.py --sanity
```

### Run in Background (auto-resume)
```bash
./run_background.sh
# Resumes the latest partial run; use --no-resume for a fresh start
tail -f bg_run_*.log              # follow live output
kill $(cat experiment_bg.pid)     # stop
```

### Run All Experiments
```bash
./run_experiments.py
```

### Run Subset
```bash
# Test specific model
./run_experiments.py --models openai/gpt-oss-20b

# Test specific configurations
./run_experiments.py --models openai/gpt-oss-20b \
                     --tp 4 8 \
                     --quantization none fp8

# Override sanity defaults
./run_experiments.py --sanity --tp 4 --input-len 16 --num-prompts 8
```

### Utilities
```bash
./experiment_utils.py check    # Verify prerequisites
./experiment_utils.py status   # Check current state
./experiment_utils.py stop     # Stop vLLM server
./experiment_utils.py backup   # Backup results
./experiment_utils.py clean    # Remove results (with confirmation)
./experiment_utils.py logs     # Tail latest logs
```

### Analysis
```bash
./analyze_results.py                          # Analyze results
./analyze_results.py --results-dir ./path     # Custom results dir
```

## Workflow Best Practices

1. **Before Starting**: Run `./experiment_utils.py check` to verify all prerequisites
2. **Backup Results**: Use `./experiment_utils.py backup` before new runs
3. **Monitor Progress**: Use `./experiment_utils.py status` in another terminal
4. **Check GPU Usage**: Run `xpu-smi stats -d 3` to monitor hardware
5. **After Completion**: Always run `./analyze_results.py` to generate reports

## Known Issues & Solutions

### Issue: Server Fails to Start
- **Cause**: Previous vLLM process still running
- **Solution**: `./experiment_utils.py stop`

### Issue: Out of Memory
- **Solution**: Reduce `--gpu-memory-util` from 0.9 to 0.8 or 0.7
- Edit `build_vllm_command()` method in `run_experiments.py`

### Issue: Benchmark Timeout
- **Cause**: Model too large or TP value too low
- **Solution**: Increase `--timeout-benchmark` or skip that configuration

### Issue: Container Not Running
- **Solution**: Check with `docker ps`, restart with `docker start vllm-test`
- If doesn't exist, create using commands in `old/official.sh`

## Development Guidelines

### Keeping Docs in Sync (REQUIRED after any non-trivial change)

After modifying any script or behaviour, update the affected docs **in the same session**:

| Change type | Files to update |
|---|---|
| New CLI flag or parameter | `copilot-instructions.md` (Configuration Parameters + Architecture Notes), `EXPERIMENT_AUTOMATION.md` (Configuration section + defaults table), `QUICK_REFERENCE.md` (examples + defaults table) |
| New script or file | `copilot-instructions.md` (repo structure tree + Architecture Notes), `QUICK_REFERENCE.md` (Files table), `EXPERIMENT_AUTOMATION.md` (Quick Start or relevant section) |
| New rule / learned correction | `copilot-instructions.md` (add numbered rule under Critical Rules & Corrections) |
| Deleted file | Remove **all** references from `copilot-instructions.md`, `EXPERIMENT_AUTOMATION.md`, `QUICK_REFERENCE.md` |
| Changed default values | `copilot-instructions.md` (Configuration Parameters), `EXPERIMENT_AUTOMATION.md` (defaults blocks), `QUICK_REFERENCE.md` (Default Parameters table) |
| Output file added/removed | `copilot-instructions.md` (repo structure tree + Rule 17), `EXPERIMENT_AUTOMATION.md` (Output Structure section) |

**`copilot-instructions.md` update checklist:**
1. Repo structure tree — add/remove/rename files
2. Critical Rules — add a new numbered rule if a bug or correction was learned
3. Configuration Parameters — update defaults
4. Common Commands — update examples
5. Architecture Notes — update component descriptions
6. Future Enhancements — mark items `[x]` when completed
7. Bump **Last Updated** line with date and a one-line summary of what changed

### When Adding New Features
1. Update `run_experiments.py` for experiment logic
2. Update `experiment_utils.py` for utility commands
3. Update `analyze_results.py` for new metrics
4. Update `EXPERIMENT_AUTOMATION.md`, `QUICK_REFERENCE.md`, and `copilot-instructions.md` (see checklist above)
5. Test with a single experiment first

### When Modifying Configurations
- Use command-line arguments instead of editing code
- Document non-obvious parameter choices
- Test with `--models openai/gpt-oss-20b --tp 2 --quantization none` first

### Code Style
- Use type hints throughout
- Prefer dataclasses for configuration
- Use descriptive variable names
- Add docstrings to all functions
- Keep methods focused and small

## Integration with User Instructions

This repository follows the user's global instructions:
- **Surface Assumptions**: When unclear about benchmark parameters, state assumptions explicitly
- **Simplicity First**: Prefer straightforward solutions over complex abstractions
- **Scope Discipline**: Don't modify unrelated code or "clean up" working scripts
- **Verification**: Always test changes with at least one experiment run
- **Self-Improvement**: Update this file when corrections are made

## Future Enhancements (Planned)

- [x] Resume capability for interrupted runs (`--resume` / `run_background.sh`)
- [ ] Parallel experiment execution (multiple GPUs)
- [ ] Real-time web dashboard for monitoring
- [ ] Automatic optimal configuration recommendation
- [ ] Integration with MLflow for experiment tracking
- [ ] Support for additional quantization methods (int8, int4)

---

## Architecture Notes

### Unified Runner (run_experiments.py)
- `VLLMExperimentRunner` is the single class for all experiments.
- Runs **inside** the Docker container; no `docker exec` wrappers.
- `exec_local(cmd)` replaces the old `docker_exec()` method.
- `run_all()` replaces the old `run_all_experiments()` method.
- Benchmark parameters (`input_len`, `output_len`, `concurrency`, `num_prompts`) are constructor args.
- Two default sets: `FULL_BENCH_DEFAULTS` and `SANITY_DEFAULTS`.
- `--sanity` CLI flag selects `SANITY_DEFAULTS`; any param can still be overridden.

---

**Last Updated**: February 25, 2026 (session 2 — added run_background.sh, --resume, --max-model-len, raw_results.json, per-model best configs)
**Primary Maintainer**: Senior Software Engineer, Intel
