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
/root/dkorat/deep-research/
├── run_experiments.py          # Main automation engine (Python)
├── run_sanity_test.py          # Minimal sanity test (4 tests, 8 token I/O, ~5 min)
├── experiment_utils.py         # Utility commands (status, stop, clean, backup, check, logs)
├── analyze_results.py          # Results analysis and visualization
├── EXPERIMENT_AUTOMATION.md    # Comprehensive documentation
├── QUICK_REFERENCE.txt         # Quick command reference
├── old/                        # Deprecated bash scripts (keep for reference)
│   ├── official.sh             # Original manual benchmark commands
│   ├── client.sh               # Original client script
│   ├── serve.sh                # Original server script
│   └── run_experiments.sh      # Old bash automation (deprecated)
├── ai-code-slop/               # Web UI components (separate project)
├── experiment_results/         # Created at runtime (full experiments)
│   ├── *_results.json          # Individual experiment results
│   ├── summary.txt             # Run summary
│   ├── detailed_analysis.txt   # Full analysis
│   ├── results_summary.csv     # Spreadsheet export
│   └── logs/                   # Server and benchmark logs
└── sanity_test_results/        # Created at runtime (sanity tests)
    └── [same structure as experiment_results]
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

### 3. Container Management
**RULE**: Always verify container is running before experiments
- Container name: `vllm-test`
- Check with: `docker ps | grep vllm-test`
- The container must be created using the commands in `old/official.sh`

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
**RULE**: Use `no_proxy=localhost,127.0.0.1,0.0.0.0` matching the container's original config
- Intel proxy intercepts connections; must explicitly bypass local addresses
- Use the exact same no_proxy values as the container was created with
- Clear HTTP_PROXY/HTTPS_PROXY vars, keep no_proxy matching official.sh

```python
# Correct implementation
result = subprocess.run(
    ["docker", "exec", "-i",
     "-e", "NO_PROXY=localhost,127.0.0.1,0.0.0.0",
     "-e", "no_proxy=localhost,127.0.0.1,0.0.0.0",
     "-e", "HTTP_PROXY=",
     "-e", "HTTPS_PROXY=",
     "-e", "http_proxy=",
     "-e", "https_proxy=",
     container_name, "bash", "-i", "-c", benchmark_cmd],
    ...
)
```

### 10. oneAPI Environment Initialization
**RULE**: Pipe commands via stdin to `docker exec -i bash` AND explicitly source `/etc/bash.bashrc`
- `docker exec -it vllm-test bash` loads oneAPI via `/etc/bash.bashrc` because bash is interactive
- `bash -c "cmd"` skips this → vLLM XPU kernels unavailable → benchmark hangs/fails
- Piping stdin without a TTY also skips `.bashrc` auto-sourcing
- ✅ CORRECT: `docker exec -i container bash` + `input="source /etc/bash.bashrc\ncmd\n"`
- ❌ WRONG: `bash -c "command"` or `bash -i -c "command"` without explicit bashrc source

```python
# Correct implementation - mirrors official.sh interactive shell workflow
shell_input = "\n".join([
    "source /etc/bash.bashrc",   # load oneAPI environment
    "export no_proxy=localhost,127.0.0.1,0.0.0.0",
    "export NO_PROXY=localhost,127.0.0.1,0.0.0.0",
    "export HTTP_PROXY=",
    "export http_proxy=",
    benchmark_cmd,
]) + "\n"
result = subprocess.run(
    ["docker", "exec", "-i", container_name, "bash"],
    input=shell_input,
    text=True,
    timeout=timeout
)
```

### 10. Health Check Implementation
**RULE**: Use curl -f and don't require non-empty response body
- The /health endpoint returns 200 OK with empty body (valid)
- Use `curl -f` to fail on HTTP errors (4xx, 5xx)
- Check only returncode == 0, NOT stdout content

```python
# Correct implementation
result = self.docker_exec(f"curl -f -s http://localhost:{self.port}/health", timeout=10)
if result.returncode == 0:  # Don't check result.stdout.strip()
    Logger.success("Server is ready!")
    return True
```

### 11. Benchmark Result Validation
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

### 12. Server Startup Monitoring
**RULE**: Monitor vllm process health during startup, show logs on failure
- Check if process is running with `pgrep -f 'vllm serve'`
- Periodically verify process hasn't died during long startups
- Display last 50 lines of server logs when startup fails
- This helps debug initialization errors immediately

```python
# Correct implementation
def wait_for_server(self) -> bool:
    time.sleep(10)  # Let process initialize
    result = self.docker_exec("pgrep -f 'vllm serve' | head -1", timeout=5)
    if result.returncode != 0 or not result.stdout.strip():
        Logger.error("vLLM process not running!")
        self._show_server_logs()
        return False
```

### 13. Benchmark Host Configuration
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

### 14. Clean Up Before Experiments
**RULE**: Always kill previous vLLM servers AND benchmark processes before starting
- Kill both `vllm serve` and `vllm bench` processes
- Benchmark processes can hang and consume resources
- Use pkill -f to match full command line

```python
# Correct implementation
def stop_vllm_server(self):
    try:
        self.docker_exec("pkill -f 'vllm serve'", capture_output=False)
    except subprocess.CalledProcessError:
        pass
    try:
        self.docker_exec("pkill -f 'vllm bench'", capture_output=False)
    except subprocess.CalledProcessError:
        pass
    time.sleep(5)
```

### 15. Timestamped Result Directories
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

## Common Commands

### Run Sanity Test (Quick Validation)
```bash
./run_sanity_test.py  # ~5 min, 3 tests, 8 token I/O, 4 prompts
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

### When Adding New Features
1. Update `run_experiments.py` for experiment logic
2. Update `experiment_utils.py` for utility commands
3. Update `analyze_results.py` for new metrics
4. Update both `EXPERIMENT_AUTOMATION.md` and `QUICK_REFERENCE.txt`
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

- [ ] Resume capability for interrupted runs
- [ ] Parallel experiment execution (multiple GPUs)
- [ ] Real-time web dashboard for monitoring
- [ ] Automatic optimal configuration recommendation
- [ ] Integration with MLflow for experiment tracking
- [ ] Support for additional quantization methods (int8, int4)

---

**Last Updated**: February 24, 2026
**Primary Maintainer**: Senior Software Engineer, Intel
