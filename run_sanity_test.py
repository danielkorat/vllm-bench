#!/usr/bin/env python3
"""
Minimal sanity test for vLLM benchmarking
Quickly tests all hyperparameter options with minimal data
"""

import subprocess
import os
import shutil
import time
import json
import argparse
import threading
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict, Optional, Tuple
from experiment_common import Color, ExperimentConfig, ExperimentResult, Logger


class VLLMSanityTest:
    """Minimal sanity test runner"""
    
    def __init__(
        self,
        container_name: str = "vllm-test",
        port: int = 8000,
        results_dir: str = "./sanity_test_results",
        timeout_startup: int = 300,
        timeout_benchmark: int = 600
    ):
        self.container_name = container_name
        self.port = port
        
        # Create timestamped subdirectory in Israel Time
        israel_tz = ZoneInfo("Asia/Jerusalem")
        timestamp = datetime.now(israel_tz).strftime("%Y%m%d_%H%M")
        base_results_dir = Path(results_dir)
        self.results_dir = base_results_dir / timestamp
        self.log_dir = self.results_dir / "logs"
        
        self.timeout_startup = timeout_startup
        self.timeout_benchmark = timeout_benchmark
        
        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        Logger.log(f"Results will be saved to: {self.results_dir}")
        
        # Tracking
        self.results: List[ExperimentResult] = []
        self._server_process: Optional[subprocess.Popen] = None
    
    def docker_exec(
        self,
        command: str,
        detached: bool = False,
        capture_output: bool = True,
        timeout: Optional[int] = None
    ) -> subprocess.CompletedProcess:
        """Execute command locally (running inside the Docker container)"""
        cmd = ["bash", "-c", command]
        if detached:
            # Fire and forget – start in background, output already redirected in cmd
            subprocess.Popen(cmd, start_new_session=True)
            return subprocess.CompletedProcess(cmd, 0)
        if capture_output:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
        else:
            return subprocess.run(cmd, timeout=timeout)
    
    def _start_server_with_logging(
        self,
        cmd: List[str],
        log_file: Path,
    ) -> subprocess.Popen:
        """Start a long-running server in the background, tee-ing output to file + terminal"""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        self._server_process = process

        def _tee(proc: subprocess.Popen, path: Path):
            with open(path, 'w') as f:
                for line in proc.stdout:
                    print(line, end='', flush=True)
                    f.write(line)
                    f.flush()

        thread = threading.Thread(target=_tee, args=(process, log_file), daemon=True)
        thread.start()
        return process

    def _run_with_logging(
        self,
        cmd: List[str],
        log_file: Path,
        timeout: Optional[int] = None
    ) -> subprocess.CompletedProcess:
        """Run subprocess while capturing output to file and printing to terminal"""
        try:
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                output_lines = []
                try:
                    for line in process.stdout:
                        # Print to terminal
                        print(line, end='', flush=True)
                        # Write to log file
                        f.write(line)
                        f.flush()
                        output_lines.append(line)
                
                    returncode = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    raise
                
                return subprocess.CompletedProcess(
                    cmd,
                    returncode,
                    stdout=''.join(output_lines),
                    stderr=''
                )
        except Exception as e:
            Logger.error(f"Error running command with logging: {e}")
            raise
    
    def check_container_running(self) -> bool:
        """Check if vLLM is available in this environment"""
        try:
            result = subprocess.run(
                ["which", "vllm"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def stop_vllm_server(self):
        """Stop any running vLLM server and benchmark processes"""
        Logger.log("Stopping any running vLLM server and benchmarks...")
        # Kill tracked server process if present
        if self._server_process is not None:
            try:
                self._server_process.kill()
                self._server_process.wait(timeout=10)
            except Exception:
                pass
            self._server_process = None
        try:
            self.docker_exec("pkill -f 'vllm serve'", capture_output=False)
        except subprocess.CalledProcessError:
            pass  # It's OK if no process to kill
        try:
            self.docker_exec("pkill -f 'vllm bench'", capture_output=False)
        except subprocess.CalledProcessError:
            pass  # It's OK if no process to kill
        time.sleep(5)
    
    def wait_for_server(self) -> bool:
        """Wait for vLLM server to be ready"""
        Logger.log(f"Waiting for vLLM server to be ready (timeout: {self.timeout_startup}s)...")
        
        # First, wait a bit for process to initialize
        time.sleep(10)
        
        # Check if vllm process is running
        try:
            result = self.docker_exec("pgrep -f 'vllm serve' | head -1", timeout=5)
            if result.returncode != 0 or not result.stdout.strip():
                Logger.error("vLLM process not running!")
                self._show_server_logs()
                return False
            Logger.log(f"vLLM process detected (PID: {result.stdout.strip()})")
        except Exception as e:
            Logger.warning(f"Failed to check process: {e}")
        
        elapsed = 10
        while elapsed < self.timeout_startup:
            try:
                result = self.docker_exec(
                    f"curl -f -s http://localhost:{self.port}/health",
                    timeout=10
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
                # Check if process is still running
                try:
                    result = self.docker_exec("pgrep -f 'vllm serve'", timeout=5)
                    if result.returncode != 0:
                        Logger.error("vLLM process died during startup!")
                        self._show_server_logs()
                        return False
                except Exception:
                    pass
        
        Logger.error(f"Server failed to start within {self.timeout_startup}s")
        self._show_server_logs()
        return False
    
    def _show_server_logs(self, lines: int = 50):
        """Server output is printed directly to the terminal."""
        Logger.log("(Server output is printed directly to this terminal above)")
    
    def build_vllm_command(self, config: ExperimentConfig) -> str:
        """Build vLLM server command"""
        cmd_parts = [
            "VLLM_WORKER_MULTIPROC_METHOD=spawn vllm serve",
            config.model,
            "--dtype=bfloat16",
            f"--port {self.port}",
            "--block-size 64",
            "--gpu-memory-util 0.9",
            "--no-enable-prefix-caching",
            "--disable-sliding-window",
            "--disable-log-requests",
            "--max-num-batched-tokens=8192",
            "--max-model-len 2048",
            f"-tp={config.tp}"
        ]
        
        if config.enforce_eager:
            cmd_parts.append("--enforce-eager")
        
        if config.quantization:
            cmd_parts.append(f"--quantization {config.quantization}")
        
        return " ".join(cmd_parts)
    
    def build_benchmark_command(self, config: ExperimentConfig) -> str:
        """Build minimal benchmark command for sanity testing"""
        return f"""vllm bench serve \
            --host 0.0.0.0 \
            --port {self.port} \
            --model {config.model} \
            --trust-remote-code \
            --dataset-name random \
            --random-input-len 8 \
            --random-output-len 8 \
            --ignore-eos \
            --max-concurrency 2 \
            --num-prompts 4 \
            --num-warmup 3 \
            --save-result --result-filename /tmp/benchmark_result.json"""
    
    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment"""
        Logger.log("=" * 72)
        Logger.log(f"Starting sanity test: {config.name}")
        Logger.log(f"Model: {config.model}")
        Logger.log(f"Tensor Parallelism: {config.tp}")
        Logger.log(f"Quantization: {config.quantization or 'none'}")
        Logger.log(f"Enforce Eager: {config.enforce_eager}")
        Logger.log("=" * 72)
        
        start_time = time.time()
        
        # Prepare log files
        server_log = self.log_dir / f"{config.name}_server.log"
        benchmark_log = self.log_dir / f"{config.name}_benchmark.log"
        result_file = self.results_dir / f"{config.name}_results.json"
        
        # Stop any existing server
        self.stop_vllm_server()
        
        # Start vLLM server
        Logger.log("Starting vLLM server...")
        vllm_cmd = self.build_vllm_command(config)
        
        try:
            # Write a startup script that sources oneAPI before launching vLLM,
            # identical to how docker exec -it bash loads /etc/bash.bashrc.
            server_script_content = "\n".join([
                "#!/bin/bash",
                "source /opt/intel/oneapi/setvars.sh --force",  # load oneAPI environment
                "export no_proxy=localhost,127.0.0.1,0.0.0.0",
                "export NO_PROXY=localhost,127.0.0.1,0.0.0.0",
                vllm_cmd,
            ]) + "\n"
            server_script = "/tmp/start_vllm_server.sh"
            with open(server_script, 'w') as f:
                f.write(server_script_content)
            os.chmod(server_script, 0o755)
            # Start server in background, tee output to log file + terminal
            Logger.log(f"Server logs will be saved to: {server_log}")
            self._start_server_with_logging(
                ["bash", "--login", server_script],
                server_log,
            )
        except Exception as e:
            Logger.error(f"Failed to start server: {e}")
            return ExperimentResult(
                config=config,
                success=False,
                error_message=f"Server start failed: {e}",
                duration=time.time() - start_time
            )
        
        # Wait for server to be ready
        if not self.wait_for_server():
            self.stop_vllm_server()
            return ExperimentResult(
                config=config,
                success=False,
                error_message="Server startup timeout",
                duration=time.time() - start_time
            )
        
        # Run benchmark
        Logger.log("Running minimal benchmark (input_len=8, output_len=8, 4 prompts)...")
        benchmark_cmd = self.build_benchmark_command(config)
        
        try:
            # Write benchmark script directly to /tmp and run with bash --login
            # so /etc/profile -> /etc/bash.bashrc -> oneAPI is loaded.
            script_content = "\n".join([
                "#!/bin/bash",
                "source /opt/intel/oneapi/setvars.sh --force",  # load oneAPI environment
                "export no_proxy=localhost,127.0.0.1,0.0.0.0",
                "export NO_PROXY=localhost,127.0.0.1,0.0.0.0",
                benchmark_cmd,
            ]) + "\n"
            Logger.log(f"Benchmark command: {benchmark_cmd}")
            Logger.log(f"Benchmark logs will be saved to: {benchmark_log}")

            bench_script = "/tmp/run_benchmark.sh"
            with open(bench_script, 'w') as f:
                f.write(script_content)
            os.chmod(bench_script, 0o755)

            result = self._run_with_logging(
                ["bash", "--login", bench_script],
                benchmark_log,
                timeout=self.timeout_benchmark
            )
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, benchmark_cmd)
            
            # Copy result file from /tmp to results directory
            src = Path("/tmp/benchmark_result.json")
            if not src.exists():
                raise Exception("Benchmark result file not found")
            shutil.copy(src, result_file)
            
            # Rewrite as pretty-printed JSON for readability
            try:
                with open(result_file) as f:
                    _raw = json.load(f)
                with open(result_file, 'w') as f:
                    json.dump(_raw, f, indent=2)
            except Exception:
                pass
            
            # Clean up temp result file
            try:
                src.unlink(missing_ok=True)
            except Exception:
                pass
            
            Logger.success(f"Sanity test completed successfully: {config.name}")
            
            # Extract key metrics if possible
            try:
                with open(result_file) as f:
                    data = json.load(f)
                # Check if benchmark actually succeeded
                failed_requests = data.get('failed', 0)
                successful_requests = data.get('completed', 0)
                
                if failed_requests > 0 or successful_requests == 0:
                    raise Exception(f"Benchmark had {failed_requests} failures, {successful_requests} successes")
                
                throughput = data.get('request_throughput', 'N/A')
                latency = data.get('mean_ttft_ms', 'N/A')
                Logger.log(f"Requests: {successful_requests}, Throughput: {throughput} req/s, TTFT: {latency} ms")
            except Exception as e:
                Logger.error(f"Benchmark validation failed: {e}")
                raise
            
            duration = time.time() - start_time
            return ExperimentResult(config=config, success=True, duration=duration)
            
        except subprocess.TimeoutExpired:
            Logger.error(f"Benchmark timed out for {config.name}")
            return ExperimentResult(
                config=config,
                success=False,
                error_message="Benchmark timeout",
                duration=time.time() - start_time
            )
        except Exception as e:
            Logger.error(f"Benchmark failed for {config.name}: {e}")
            return ExperimentResult(
                config=config,
                success=False,
                error_message=f"Benchmark failed: {e}",
                duration=time.time() - start_time
            )
        finally:
            self.stop_vllm_server()
            time.sleep(5)  # Brief pause between experiments
    
    def run_all_tests(
        self,
        models: List[str],
        tp_values: List[int],
        quantization_options: List[Optional[str]],
        enforce_eager_options: List[bool]
    ):
        """Run all test combinations"""
        
        # Check environment
        if not self.check_container_running():
            Logger.error("vLLM command not found in PATH!")
            Logger.log("Please ensure vLLM is installed and oneAPI environment is loaded")
            return False
        
        # Generate all configurations
        configs = []
        for model in models:
            for tp in tp_values:
                for quant in quantization_options:
                    for eager in enforce_eager_options:
                        # Skip fp8 + eager=false (known to fail)
                        if quant == 'fp8' and not eager:
                            Logger.warning(f"Skipping {model} tp={tp} quant=fp8 eager=false (known failure)")
                            continue
                        configs.append(ExperimentConfig(
                            model=model,
                            tp=tp,
                            quantization=quant,
                            enforce_eager=eager
                        ))
        
        Logger.log(f"Starting vLLM sanity tests")
        Logger.log(f"Results will be saved to: {self.results_dir}")
        Logger.log(f"Total tests to run: {len(configs)}")
        Logger.log("Parameters: input_len=8, output_len=8, concurrency=2, prompts=4")
        
        start_time = time.time()
        
        # Run all tests
        for i, config in enumerate(configs, 1):
            Logger.log(f"\nTest {i}/{len(configs)}")
            result = self.run_experiment(config)
            self.results.append(result)
        
        duration = time.time() - start_time
        
        # Generate summary
        self.generate_summary(duration)
        
        return all(r.success for r in self.results)
    
    def generate_summary(self, total_duration: float):
        """Generate summary report"""
        Logger.log("=" * 72)
        Logger.log("Generating sanity test summary...")
        Logger.log("=" * 72)
        
        total_runs = len(self.results)
        successful_runs = sum(1 for r in self.results if r.success)
        failed_runs = total_runs - successful_runs
        success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
        
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        seconds = int(total_duration % 60)
        
        summary_file = self.results_dir / "summary.txt"
        
        with open(summary_file, 'w') as f:
            lines = [
                "vLLM Sanity Test Summary",
                "=" * 72,
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "Test Parameters:",
                "  - Input Length: 8 tokens",
                "  - Output Length: 8 tokens",
                "  - Concurrency: 2",
                "  - Num Prompts: 4",
                "",
                f"Total Tests: {total_runs}",
                f"Successful: {successful_runs}",
                f"Failed: {failed_runs}",
                f"Success Rate: {success_rate:.2f}%",
                "",
                f"Total Duration: {hours}h {minutes}m {seconds}s",
                ""
            ]
            
            successful = [r for r in self.results if r.success]
            if successful:
                lines.append("Successful Tests:")
                lines.append("-" * 72)
                for result in successful:
                    lines.append(f"  ✓ {result.config.name} ({result.duration:.1f}s)")
                lines.append("")
            
            failed = [r for r in self.results if not r.success]
            if failed:
                lines.append("Failed Tests:")
                lines.append("-" * 72)
                for result in failed:
                    lines.append(f"  ✗ {result.config.name} ({result.error_message})")
                lines.append("")
            
            lines.extend([
                f"Results Location: {self.results_dir}",
                f"Logs Location: {self.log_dir}"
            ])
            
            summary_text = "\n".join(lines)
            f.write(summary_text)
            print(summary_text)
        
        Logger.success(f"Summary saved to: {summary_file}")
        
        if failed_runs == 0:
            Logger.success("All sanity tests passed! ✓")
        else:
            Logger.warning("Some tests failed. Check logs for details.")
        
        # Automatically run analysis on completed results
        Logger.log("Running analysis on results...")
        try:
            import analyze_results as ar
            ar.analyze_results(str(self.results_dir))
        except Exception as e:
            Logger.warning(f"Analysis failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Minimal vLLM sanity test - covers all hyperparameters quickly'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=[
            'Qwen/Qwen3-4B-Thinking-2507'
        ],
        help='Models to test (default: 4B model for speed)'
    )
    parser.add_argument(
        '--tp',
        nargs='+',
        type=int,
        default=[2],
        help='Tensor parallelism values (default: 2 only for speed)'
    )
    parser.add_argument(
        '--quantization',
        nargs='+',
        default=['none', 'fp8'],
        help='Quantization modes (default: all modes to test coverage)'
    )
    parser.add_argument(
        '--enforce-eager',
        nargs='+',
        choices=['true', 'false'],
        default=['true', 'false'],
        help='Enforce eager execution modes to test (default: both)'
    )
    parser.add_argument(
        '--container',
        default='vllm-test',
        help='Docker container name'
    )
    parser.add_argument(
        '--results-dir',
        default='./sanity_test_results',
        help='Results directory'
    )
    parser.add_argument(
        '--timeout-startup',
        type=int,
        default=180,
        help='Server startup timeout in seconds'
    )
    parser.add_argument(
        '--timeout-benchmark',
        type=int,
        default=300,
        help='Benchmark timeout in seconds (default: 300 for minimal test)'
    )
    
    args = parser.parse_args()
    
    # Convert 'none' to None for quantization
    quant_options = [None if q == 'none' else q for q in args.quantization]
    
    # Convert eager strings to booleans
    eager_options = [e == 'true' for e in args.enforce_eager]
    
    tester = VLLMSanityTest(
        container_name=args.container,
        results_dir=args.results_dir,
        timeout_startup=args.timeout_startup,
        timeout_benchmark=args.timeout_benchmark
    )
    
    success = tester.run_all_tests(
        models=args.models,
        tp_values=args.tp,
        quantization_options=quant_options,
        enforce_eager_options=eager_options
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
