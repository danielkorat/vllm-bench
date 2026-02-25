#!/usr/bin/env python3
"""
Automated vLLM benchmarking experiment runner
Runs all combinations of models, tensor parallelism, quantization, and eager mode
"""

import subprocess
import tempfile
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict, Optional, Tuple
from experiment_common import Color, ExperimentConfig, ExperimentResult, Logger


class VLLMExperimentRunner:
    """Main experiment runner"""
    
    def __init__(
        self,
        container_name: str = "vllm-test",
        port: int = 8000,
        results_dir: str = "./experiment_results",
        timeout_startup: int = 300,
        timeout_benchmark: int = 1800
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
    
    def docker_exec(
        self,
        command: str,
        detached: bool = False,
        capture_output: bool = True,
        timeout: Optional[int] = None
    ) -> subprocess.CompletedProcess:
        """Execute command in Docker container"""
        cmd = ["docker", "exec"]
        if detached:
            cmd.append("-d")
        cmd.extend([self.container_name, "bash", "-c", command])
        
        if capture_output:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
        else:
            return subprocess.run(cmd, timeout=timeout)
    
    def check_container_running(self) -> bool:
        """Check if container is running"""
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return self.container_name in result.stdout
        except Exception:
            return False
    
    def stop_vllm_server(self):
        """Stop any running vLLM server and benchmark processes"""
        Logger.log("Stopping any running vLLM server and benchmarks...")
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
                        return False
                except Exception:
                    pass
        
        Logger.error(f"Server failed to start within {self.timeout_startup}s")
        return False
    
    
    def _run_in_container_with_logging(
        self,
        script_path: str,
        log_file: Path,
        timeout: Optional[int] = None
    ) -> Tuple[int, str]:
        """Run command in container and capture output to file + terminal"""
        result = subprocess.run(
            ["docker", "exec", self.container_name, "bash", "--login", script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout
        )
        
        # Save output to log file
        with open(log_file, 'w') as f:
            f.write(result.stdout)
        
        # Also print to terminal
        if result.stdout.strip():
            print(result.stdout)
        
        return result.returncode, result.stdout
    
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
            "--trust-remote-code",
            "--disable-sliding-window",
            "--disable-log-requests",
            "--max-num-batched-tokens=8192",
            "--max-model-len 16384",
            f"-tp={config.tp}"
        ]
        
        if config.enforce_eager:
            cmd_parts.append("--enforce-eager")
        
        if config.quantization:
            cmd_parts.append(f"--quantization {config.quantization}")
        
        return " ".join(cmd_parts)
    
    def build_benchmark_command(self, config: ExperimentConfig) -> str:
        """Build benchmark command"""
        return f"""vllm bench serve \
            --host 0.0.0.0 \
            --port {self.port} \
            --model {config.model} \
            --trust-remote-code \
            --dataset-name random \
            --random-input-len 1024 \
            --random-output-len 1024 \
            --ignore-eos \
            --max-concurrency 32 \
            --num-prompts 160 \
            --num-warmup 3 \
            --save-result --result-filename /tmp/benchmark_result.json"""
    
    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment"""
        Logger.log("=" * 72)
        Logger.log(f"Starting experiment: {config.name}")
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
        Logger.log(f"Server logs will be saved to: {server_log}")
        vllm_cmd = self.build_vllm_command(config)
        
        try:
            self.docker_exec(
                f"{vllm_cmd} > /tmp/vllm_server.log 2>&1",
                detached=True
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
            # Save server logs
            try:
                result = self.docker_exec("cat /tmp/vllm_server.log")
                server_log.write_text(result.stdout)
                Logger.log(f"Server logs saved to: {server_log}")
                # Also print for debugging
                if result.stdout.strip():
                    Logger.error("=" * 72)
                    Logger.error("Server output:")
                    Logger.error("-" * 72)
                    print(result.stdout)
                    Logger.error("=" * 72)
            except Exception:
                pass
            
            self.stop_vllm_server()
            return ExperimentResult(
                config=config,
                success=False,
                error_message="Server startup timeout",
                duration=time.time() - start_time
            )
        
        # Save server logs
        try:
            result = self.docker_exec("cat /tmp/vllm_server.log")
            server_log.write_text(result.stdout)
            Logger.success(f"Server logs saved to: {server_log}")
        except Exception:
            pass
        
        # Run benchmark
        Logger.log("Running benchmark...")
        Logger.log(f"Benchmark logs will be saved to: {benchmark_log}")
        benchmark_cmd = self.build_benchmark_command(config)
        
        try:
            # Write benchmark to a shell script, copy into container, run with
            # bash --login so /etc/profile -> /etc/bash.bashrc -> oneAPI is loaded.
            # This exactly mirrors what happens in: docker exec -it vllm-test bash
            script_content = "\n".join([
                "#!/bin/bash",
                "source /etc/bash.bashrc",  # load oneAPI environment
                "export no_proxy=localhost,127.0.0.1,0.0.0.0",
                "export NO_PROXY=localhost,127.0.0.1,0.0.0.0",
                "export HTTP_PROXY=",
                "export HTTPS_PROXY=",
                "export http_proxy=",
                "export https_proxy=",
                benchmark_cmd,
            ]) + "\n"
            Logger.log(f"Benchmark command: {benchmark_cmd}")

            with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
                f.write(script_content)
                local_script = f.name
            try:
                subprocess.run(
                    ["docker", "cp", local_script, f"{self.container_name}:/tmp/run_benchmark.sh"],
                    check=True, timeout=10
                )
            finally:
                os.unlink(local_script)

            subprocess.run(
                ["docker", "exec", self.container_name, "chmod", "+x", "/tmp/run_benchmark.sh"],
                check=True, timeout=5
            )

            # Run benchmark and capture output
            returncode, stdout = self._run_in_container_with_logging(
                "/tmp/run_benchmark.sh",
                benchmark_log,
                timeout=self.timeout_benchmark
            )
            
            if returncode != 0:
                raise subprocess.CalledProcessError(returncode, benchmark_cmd)
            
            # Copy result file from container
            copy_result = subprocess.run(
                ["docker", "cp", f"{self.container_name}:/tmp/benchmark_result.json", str(result_file)],
                capture_output=True,
                timeout=30
            )
            
            if copy_result.returncode != 0:
                raise Exception("Benchmark result file not found")
            
            # Rewrite as pretty-printed JSON for readability
            try:
                with open(result_file) as f:
                    _raw = json.load(f)
                with open(result_file, 'w') as f:
                    json.dump(_raw, f, indent=2)
            except Exception:
                pass
            
            # Clean up container temp file
            try:
                self.docker_exec("rm -f /tmp/benchmark_result.json")
            except Exception:
                pass
            
            Logger.success(f"Benchmark logs saved to: {benchmark_log}")
            Logger.success(f"Experiment completed successfully: {config.name}")
            
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
            time.sleep(10)  # Brief pause between experiments
    
    def run_all_experiments(
        self,
        models: List[str],
        tp_values: List[int],
        quantization_options: List[Optional[str]],
        enforce_eager_options: List[bool]
    ):
        """Run all experiment combinations"""
        
        # Check container
        if not self.check_container_running():
            Logger.error(f"Container {self.container_name} is not running!")
            Logger.log("Please start the container first")
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
        
        Logger.log(f"Starting automated vLLM experiments")
        Logger.log(f"Results will be saved to: {self.results_dir}")
        Logger.log(f"Total experiments to run: {len(configs)}")
        
        start_time = time.time()
        
        # Run all experiments
        for i, config in enumerate(configs, 1):
            Logger.log(f"\nExperiment {i}/{len(configs)}")
            result = self.run_experiment(config)
            self.results.append(result)
        
        duration = time.time() - start_time
        
        # Generate summary
        self.generate_summary(duration)
        
        return all(r.success for r in self.results)
    
    def generate_summary(self, total_duration: float):
        """Generate summary report"""
        Logger.log("=" * 72)
        Logger.log("Generating summary report...")
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
                "vLLM Benchmark Experiment Summary",
                "=" * 72,
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                f"Total Experiments: {total_runs}",
                f"Successful: {successful_runs}",
                f"Failed: {failed_runs}",
                f"Success Rate: {success_rate:.2f}%",
                "",
                f"Total Duration: {hours}h {minutes}m {seconds}s",
                ""
            ]
            
            successful = [r for r in self.results if r.success]
            if successful:
                lines.append("Successful Experiments:")
                lines.append("-" * 72)
                for result in successful:
                    lines.append(f"  ✓ {result.config.name}")
                lines.append("")
            
            failed = [r for r in self.results if not r.success]
            if failed:
                lines.append("Failed Experiments:")
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
            Logger.success("All experiments succeeded! 🎉")
        else:
            Logger.warning("Some experiments failed. Check logs for details.")
        
        # Automatically run analysis on completed results
        Logger.log("Running analysis on results...")
        try:
            import analyze_results as ar
            ar.analyze_results(str(self.results_dir))
        except Exception as e:
            Logger.warning(f"Analysis failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Automated vLLM benchmarking experiments'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=[
            'openai/gpt-oss-20b',
            'Qwen/Qwen3-30B-A3B',
            'Qwen/Qwen3-4B-Thinking-2507'
        ],
        help='Models to benchmark'
    )
    parser.add_argument(
        '--tp',
        nargs='+',
        type=int,
        default=[2, 4, 8],
        help='Tensor parallelism values'
    )
    parser.add_argument(
        '--quantization',
        nargs='+',
        default=['none', 'fp8'],
        help='Quantization modes (use "none" for no quantization)'
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
        help='Results directory'
    )
    parser.add_argument(
        '--timeout-startup',
        type=int,
        default=300,
        help='Server startup timeout in seconds'
    )
    parser.add_argument(
        '--timeout-benchmark',
        type=int,
        default=1800,
        help='Benchmark timeout in seconds'
    )
    
    args = parser.parse_args()
    
    # Convert 'none' to None for quantization
    quant_options = [None if q == 'none' else q for q in args.quantization]
    
    # Convert eager strings to booleans
    eager_options = [e == 'true' for e in args.enforce_eager]
    
    runner = VLLMExperimentRunner(
        container_name=args.container,
        results_dir=args.results_dir,
        timeout_startup=args.timeout_startup,
        timeout_benchmark=args.timeout_benchmark
    )
    
    success = runner.run_all_experiments(
        models=args.models,
        tp_values=args.tp,
        quantization_options=quant_options,
        enforce_eager_options=eager_options
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
