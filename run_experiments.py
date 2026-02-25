#!/usr/bin/env python3
"""
Automated vLLM benchmarking experiment runner.

Runs combinatorial benchmarks across models, tensor-parallelism, quantization
and eager-mode settings.

Designed to run *inside* the Docker container (intel/vllm:0.14.1-xpu).
All commands are executed locally via bash – no docker exec wrappers needed.

Usage:
    ./run_experiments.py                     # full benchmark suite
    ./run_experiments.py --sanity            # quick sanity check (small defaults)
    ./run_experiments.py --models Qwen/...   # override any parameter
"""

import subprocess
import shutil
import os
import time
import json
import argparse
import threading
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Optional
from experiment_common import ExperimentConfig, ExperimentResult, Logger

# ---------------------------------------------------------------------------
# Default parameter sets
# ---------------------------------------------------------------------------

FULL_BENCH_DEFAULTS: dict = dict(
    models=[
        'openai/gpt-oss-20b',
        'Qwen/Qwen3-30B-A3B',
        'Qwen/Qwen3-4B-Thinking-2507',
    ],
    tp=[2, 4, 8],
    quantization=['none', 'fp8'],
    enforce_eager=['true', 'false'],
    results_dir='./experiment_results',
    timeout_startup=300,
    timeout_benchmark=1800,
    input_len=1024,
    output_len=1024,
    concurrency=32,
    num_prompts=160,
)

SANITY_DEFAULTS: dict = dict(
    models=['Qwen/Qwen3-4B-Thinking-2507'],
    tp=[2],
    quantization=['none', 'fp8'],
    enforce_eager=['true', 'false'],
    results_dir='./sanity_test_results',
    timeout_startup=180,
    timeout_benchmark=300,
    input_len=8,
    output_len=8,
    concurrency=2,
    num_prompts=4,
)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class VLLMExperimentRunner:
    """
    Experiment runner for vLLM benchmarks.

    Runs *inside* the Docker container – all commands are executed locally via
    ``bash``.  The caller is expected to be the process running inside the
    container (e.g. launched via ``docker exec -it vllm-test bash``).
    """

    def __init__(
        self,
        port: int = 8000,
        results_dir: str = './experiment_results',
        timeout_startup: int = 300,
        timeout_benchmark: int = 1800,
        input_len: int = 1024,
        output_len: int = 1024,
        concurrency: int = 32,
        num_prompts: int = 160,
    ):
        self.port = port
        self.timeout_startup = timeout_startup
        self.timeout_benchmark = timeout_benchmark
        self.input_len = input_len
        self.output_len = output_len
        self.concurrency = concurrency
        self.num_prompts = num_prompts

        # Create timestamped subdirectory in Israel Time
        israel_tz = ZoneInfo("Asia/Jerusalem")
        timestamp = datetime.now(israel_tz).strftime("%Y%m%d_%H%M")
        self.results_dir = Path(results_dir) / timestamp
        self.log_dir = self.results_dir / "logs"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

        Logger.log(f"Results will be saved to: {self.results_dir}")

        self.results: List[ExperimentResult] = []
        self._server_process: Optional[subprocess.Popen] = None

    # ------------------------------------------------------------------
    # Low-level execution helpers
    # ------------------------------------------------------------------

    def exec_local(
        self,
        command: str,
        capture_output: bool = True,
        timeout: Optional[int] = None,
    ) -> subprocess.CompletedProcess:
        """Run a shell command locally (we are inside the container)."""
        cmd = ["bash", "-c", command]
        if capture_output:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        else:
            return subprocess.run(cmd, timeout=timeout)

    def _start_server_with_logging(
        self,
        cmd: List[str],
        log_file: Path,
    ) -> subprocess.Popen:
        """Start the vLLM server in the background, tee-ing output to file + terminal."""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._server_process = process

        def _tee(proc: subprocess.Popen, path: Path):
            with open(path, 'w') as f:
                for line in proc.stdout:
                    print(line, end='', flush=True)
                    f.write(line)
                    f.flush()

        threading.Thread(target=_tee, args=(process, log_file), daemon=True).start()
        return process

    def _run_with_logging(
        self,
        cmd: List[str],
        log_file: Path,
        timeout: Optional[int] = None,
    ) -> subprocess.CompletedProcess:
        """Run a command, printing output to terminal and saving to log_file."""
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            output_lines: List[str] = []
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    f.write(line)
                    f.flush()
                    output_lines.append(line)
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                raise
        return subprocess.CompletedProcess(cmd, returncode, stdout=''.join(output_lines), stderr='')

    # ------------------------------------------------------------------
    # vLLM lifecycle
    # ------------------------------------------------------------------

    def check_environment(self) -> bool:
        """Check that vLLM is available in PATH (i.e. we are inside the container)."""
        try:
            result = subprocess.run(["which", "vllm"], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False

    def stop_vllm_server(self):
        """Kill any running vLLM server and benchmark processes."""
        Logger.log("Stopping any running vLLM server and benchmarks...")
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

    def wait_for_server(self) -> bool:
        """Poll the health endpoint until the server is ready or timeout is reached."""
        Logger.log(f"Waiting for vLLM server to be ready (timeout: {self.timeout_startup}s)...")
        time.sleep(10)

        # Verify process started
        try:
            result = self.exec_local("pgrep -f 'vllm serve' | head -1", timeout=5)
            if result.returncode != 0 or not result.stdout.strip():
                Logger.error("vLLM process not running!")
                Logger.log("(Server output is printed directly to this terminal above)")
                return False
            Logger.log(f"vLLM process detected (PID: {result.stdout.strip()})")
        except Exception as e:
            Logger.warning(f"Failed to check process: {e}")

        elapsed = 10
        while elapsed < self.timeout_startup:
            try:
                result = self.exec_local(
                    f"curl -f -s http://localhost:{self.port}/health", timeout=10
                )
                if result.returncode == 0:
                    Logger.success("Server is ready!")
                    return True
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                pass

            time.sleep(5)
            elapsed += 5

            if elapsed % 30 == 0:
                Logger.log(f"Still waiting... ({elapsed}s elapsed)")
                try:
                    result = self.exec_local("pgrep -f 'vllm serve'", timeout=5)
                    if result.returncode != 0:
                        Logger.error("vLLM process died during startup!")
                        Logger.log("(Server output is printed directly to this terminal above)")
                        return False
                except Exception:
                    pass

        Logger.error(f"Server failed to start within {self.timeout_startup}s")
        return False

    # ------------------------------------------------------------------
    # Command builders
    # ------------------------------------------------------------------

    def build_vllm_command(self, config: ExperimentConfig) -> str:
        """Build the 'vllm serve' command for the given config."""
        cmd_parts = [
            "VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve",
            config.model,
            "--dtype=bfloat16",
            f"--port {self.port}",
            "--block-size 64",
            "--gpu-memory-util 0.9",
            "--no-enable-prefix-caching",
            "--trust-remote-code",
            "--disable-sliding-window",
            "--disable-log-requests",
            "--max-num-batched-tokens=8192",
            "--max-model-len 2048", # TODO: make this configurable?
            f"-tp={config.tp}"
        ]
        
        if config.enforce_eager:
            cmd_parts.append("--enforce-eager")
        
        if config.quantization:
            cmd_parts.append(f"--quantization {config.quantization}")
        
        return " ".join(cmd_parts)

    def build_benchmark_command(self, config: ExperimentConfig) -> str:
        """Build the 'vllm bench serve' command for the given config."""
        return (
            f"vllm bench serve"
            f" --host 0.0.0.0"
            f" --port {self.port}"
            f" --model {config.model}"
            f" --trust-remote-code"
            f" --dataset-name random"
            f" --random-input-len {self.input_len}"
            f" --random-output-len {self.output_len}"
            f" --ignore-eos"
            f" --max-concurrency {self.concurrency}"
            f" --num-prompts {self.num_prompts}"
            f" --num-warmup 3"
            f" --save-result --result-filename /tmp/benchmark_result.json"
        )
    
    # ------------------------------------------------------------------
    # Experiment execution
    # ------------------------------------------------------------------

    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment (server + benchmark) for the given config."""
        Logger.log("=" * 72)
        Logger.log(f"Starting experiment: {config.name}")
        Logger.log(f"  Model:            {config.model}")
        Logger.log(f"  Tensor Parallel:  {config.tp}")
        Logger.log(f"  Quantization:     {config.quantization or 'none'}")
        Logger.log(f"  Enforce Eager:    {config.enforce_eager}")
        Logger.log(f"  Input/Output len: {self.input_len}/{self.output_len}")
        Logger.log(f"  Concurrency:      {self.concurrency}  Prompts: {self.num_prompts}")
        Logger.log("=" * 72)

        start_time = time.time()
        server_log = self.log_dir / f"{config.name}_server.log"
        benchmark_log = self.log_dir / f"{config.name}_benchmark.log"
        result_file = self.results_dir / f"{config.name}_results.json"

        self.stop_vllm_server()

        # ---- Start vLLM server ----
        Logger.log("Starting vLLM server...")
        Logger.log(f"Server logs: {server_log}")
        vllm_cmd = self.build_vllm_command(config)
        try:
            server_script = "/tmp/start_vllm_server.sh"
            with open(server_script, 'w') as f:
                f.write("\n".join([
                    "#!/bin/bash",
                    "source /opt/intel/oneapi/setvars.sh --force",
                    "export no_proxy=localhost,127.0.0.1,0.0.0.0",
                    "export NO_PROXY=localhost,127.0.0.1,0.0.0.0",
                    vllm_cmd,
                ]) + "\n")
            os.chmod(server_script, 0o755)
            self._start_server_with_logging(["bash", "--login", server_script], server_log)
        except Exception as e:
            Logger.error(f"Failed to start server: {e}")
            return ExperimentResult(
                config=config, success=False,
                error_message=f"Server start failed: {e}",
                duration=time.time() - start_time,
            )

        # ---- Wait for server ----
        if not self.wait_for_server():
            self.stop_vllm_server()
            return ExperimentResult(
                config=config, success=False,
                error_message="Server startup timeout",
                duration=time.time() - start_time,
            )

        # ---- Run benchmark ----
        Logger.log("Running benchmark...")
        Logger.log(f"Benchmark logs: {benchmark_log}")
        benchmark_cmd = self.build_benchmark_command(config)
        Logger.log(f"Benchmark command: {benchmark_cmd}")

        try:
            bench_script = "/tmp/run_benchmark.sh"
            with open(bench_script, 'w') as f:
                f.write("\n".join([
                    "#!/bin/bash",
                    "source /opt/intel/oneapi/setvars.sh --force",
                    "export no_proxy=localhost,127.0.0.1,0.0.0.0",
                    "export NO_PROXY=localhost,127.0.0.1,0.0.0.0",
                    benchmark_cmd,
                ]) + "\n")
            os.chmod(bench_script, 0o755)

            result = self._run_with_logging(
                ["bash", "--login", bench_script],
                benchmark_log,
                timeout=self.timeout_benchmark,
            )
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, benchmark_cmd)

            # Copy result from /tmp to results dir
            src = Path("/tmp/benchmark_result.json")
            if not src.exists():
                raise Exception("Benchmark result file not found")
            shutil.copy(src, result_file)
            src.unlink(missing_ok=True)

            # Pretty-print JSON
            try:
                with open(result_file) as f:
                    raw = json.load(f)
                with open(result_file, 'w') as f:
                    json.dump(raw, f, indent=2)
            except Exception:
                pass

            # Validate benchmark succeeded
            with open(result_file) as f:
                data = json.load(f)
            failed_req = data.get('failed', 0)
            ok_req = data.get('completed', 0)
            if failed_req > 0 or ok_req == 0:
                raise Exception(f"Benchmark had {failed_req} failures, {ok_req} successes")

            throughput = data.get('request_throughput', 'N/A')
            latency = data.get('mean_ttft_ms', 'N/A')
            Logger.success(f"Experiment completed: {config.name}")
            Logger.log(f"Requests: {ok_req}, Throughput: {throughput} req/s, TTFT: {latency} ms")
            return ExperimentResult(config=config, success=True, duration=time.time() - start_time)

        except subprocess.TimeoutExpired:
            Logger.error(f"Benchmark timed out for {config.name}")
            return ExperimentResult(
                config=config, success=False,
                error_message="Benchmark timeout",
                duration=time.time() - start_time,
            )
        except Exception as e:
            Logger.error(f"Benchmark failed for {config.name}: {e}")
            return ExperimentResult(
                config=config, success=False,
                error_message=f"Benchmark failed: {e}",
                duration=time.time() - start_time
            )
        finally:
            self.stop_vllm_server()
            time.sleep(5)

    def run_all(
        self,
        models: List[str],
        tp_values: List[int],
        quantization_options: List[Optional[str]],
        enforce_eager_options: List[bool],
    ) -> bool:
        """Run all experiment combinations, skipping known-bad configs."""
        if not self.check_environment():
            Logger.error("vLLM command not found in PATH!")
            Logger.log("Ensure you are running inside the container with oneAPI loaded.")
            return False

        configs = []
        for model in models:
            for tp in tp_values:
                for quant in quantization_options:
                    for eager in enforce_eager_options:
                        # Skip fp8 + eager=false (known vLLM engine failure)
                        if quant == 'fp8' and not eager:
                            Logger.warning(
                                f"Skipping {model} tp={tp} quant=fp8 eager=false (known failure)"
                            )
                            continue
                        configs.append(ExperimentConfig(
                            model=model, tp=tp, quantization=quant, enforce_eager=eager
                        ))

        Logger.log(f"Starting vLLM experiments — {len(configs)} configurations")
        Logger.log(f"Results: {self.results_dir}")
        start_time = time.time()

        for i, config in enumerate(configs, 1):
            Logger.log(f"\nExperiment {i}/{len(configs)}")
            self.results.append(self.run_experiment(config))

        self.generate_summary(time.time() - start_time)
        return all(r.success for r in self.results)

    def generate_summary(self, total_duration: float):
        """Write and print a summary report of all results."""
        Logger.log("=" * 72)
        total = len(self.results)
        ok = sum(1 for r in self.results if r.success)
        fail = total - ok
        rate = (ok / total * 100) if total else 0
        h = int(total_duration // 3600)
        m = int((total_duration % 3600) // 60)
        s = int(total_duration % 60)

        lines = [
            "vLLM Benchmark Summary",
            "=" * 72,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"  Benchmark params: input={self.input_len} output={self.output_len}"
            f"  concurrency={self.concurrency}  prompts={self.num_prompts}",
            "",
            f"Total Experiments: {total}",
            f"Successful:        {ok}",
            f"Failed:            {fail}",
            f"Success Rate:      {rate:.2f}%",
            "",
            f"Total Duration: {h}h {m}m {s}s",
            "",
        ]

        successful = [r for r in self.results if r.success]
        if successful:
            lines += ["Successful Experiments:", "-" * 72]
            lines += [f"  ✓ {r.config.name}  ({r.duration:.1f}s)" for r in successful]
            lines.append("")

        failed = [r for r in self.results if not r.success]
        if failed:
            lines += ["Failed Experiments:", "-" * 72]
            lines += [f"  ✗ {r.config.name}  ({r.error_message})" for r in failed]
            lines.append("")

        lines += [f"Results: {self.results_dir}", f"Logs:    {self.log_dir}"]

        summary_text = "\n".join(lines)
        summary_file = self.results_dir / "summary.txt"
        summary_file.write_text(summary_text)
        print(summary_text)
        Logger.success(f"Summary saved to: {summary_file}")

        if fail == 0:
            Logger.success("All experiments succeeded! 🎉")
        else:
            Logger.warning("Some experiments failed. Check logs for details.")

        Logger.log("Running analysis on results...")
        try:
            import analyze_results as ar
            ar.analyze_results(str(self.results_dir))
        except Exception as e:
            Logger.warning(f"Analysis failed: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Automated vLLM benchmarking experiments (runs inside the container)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--sanity', action='store_true',
        help='Run minimal sanity test with small defaults '
             '(single small model, tp=2, 8-token I/O, 4 prompts)',
    )
    parser.add_argument('--models', nargs='+', help='Models to benchmark')
    parser.add_argument('--tp', nargs='+', type=int, help='Tensor parallelism values')
    parser.add_argument(
        '--quantization', nargs='+',
        help='Quantization modes; use "none" to omit the --quantization flag',
    )
    parser.add_argument(
        '--enforce-eager', nargs='+', choices=['true', 'false'],
        help='Eager execution modes to test',
    )
    parser.add_argument('--results-dir', help='Results base directory')
    parser.add_argument('--timeout-startup', type=int, help='Server startup timeout (s)')
    parser.add_argument('--timeout-benchmark', type=int, help='Benchmark timeout (s)')
    parser.add_argument('--input-len', type=int, help='Random input token length')
    parser.add_argument('--output-len', type=int, help='Random output token length')
    parser.add_argument('--concurrency', type=int, help='Max concurrent requests')
    parser.add_argument('--num-prompts', type=int, help='Total number of prompts')
    parser.add_argument('--port', type=int, default=8000, help='vLLM server port')

    args = parser.parse_args()

    # Select default set based on mode
    D = SANITY_DEFAULTS if args.sanity else FULL_BENCH_DEFAULTS

    def get(attr: str, key: str):
        val = getattr(args, attr, None)
        return val if val is not None else D[key]

    quant_options = [None if q == 'none' else q for q in get('quantization', 'quantization')]
    eager_options = [e == 'true' for e in get('enforce_eager', 'enforce_eager')]

    runner = VLLMExperimentRunner(
        port=args.port,
        results_dir=get('results_dir', 'results_dir'),
        timeout_startup=get('timeout_startup', 'timeout_startup'),
        timeout_benchmark=get('timeout_benchmark', 'timeout_benchmark'),
        input_len=get('input_len', 'input_len'),
        output_len=get('output_len', 'output_len'),
        concurrency=get('concurrency', 'concurrency'),
        num_prompts=get('num_prompts', 'num_prompts'),
    )

    success = runner.run_all(
        models=get('models', 'models'),
        tp_values=get('tp', 'tp'),
        quantization_options=quant_options,
        enforce_eager_options=eager_options,
    )
    return 0 if success else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
