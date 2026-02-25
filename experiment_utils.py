#!/usr/bin/env python3
"""
Utility commands for managing vLLM experiment automation
"""

import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime
import shutil


class Color:
    """ANSI color codes"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'


class Logger:
    """Colored logging"""
    
    @staticmethod
    def timestamp() -> str:
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    @staticmethod
    def log(message: str):
        print(f"{Color.BLUE}[{Logger.timestamp()}]{Color.NC} {message}")
    
    @staticmethod
    def success(message: str):
        print(f"{Color.GREEN}[{Logger.timestamp()}] ✓{Color.NC} {message}")
    
    @staticmethod
    def error(message: str):
        print(f"{Color.RED}[{Logger.timestamp()}] ✗{Color.NC} {message}")
    
    @staticmethod
    def warning(message: str):
        print(f"{Color.YELLOW}[{Logger.timestamp()}] ⚠{Color.NC} {message}")


class ExperimentUtils:
    """Utility functions for experiment management"""
    
    def __init__(self, container_name: str = "vllm-test", results_dir: str = "./experiment_results"):
        self.container_name = container_name
        self.results_dir = Path(results_dir)
    
    def run_cmd(self, cmd: list, capture: bool = True, timeout: int = 30) -> subprocess.CompletedProcess:
        """Run a shell command"""
        return subprocess.run(
            cmd,
            capture_output=capture,
            text=True,
            timeout=timeout
        )
    
    def docker_exec(self, command: str) -> subprocess.CompletedProcess:
        """Execute command in Docker container"""
        return self.run_cmd(["docker", "exec", self.container_name, "bash", "-c", command])
    
    def is_container_running(self) -> bool:
        """Check if container is running"""
        try:
            result = self.run_cmd(
                ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"]
            )
            return self.container_name in result.stdout
        except Exception:
            return False
    
    def is_vllm_running(self) -> bool:
        """Check if vLLM server is running"""
        try:
            result = self.docker_exec("pgrep -f 'vllm serve'")
            return result.returncode == 0
        except Exception:
            return False
    
    def cmd_status(self):
        """Show current status"""
        Logger.log("Checking experiment status...")
        print()
        
        # Check container
        if self.is_container_running():
            Logger.success(f"Container {self.container_name} is running")
        else:
            Logger.error(f"Container {self.container_name} is NOT running")
        
        # Check vLLM
        if self.is_vllm_running():
            Logger.warning("vLLM server is currently running")
            print()
            try:
                result = self.docker_exec("ps aux | grep 'vllm serve' | grep -v grep")
                print(result.stdout)
            except Exception:
                pass
        else:
            Logger.success("No vLLM server running")
        
        # Check results
        if self.results_dir.exists():
            result_files = list(self.results_dir.glob("*_results.json"))
            log_files = list((self.results_dir / "logs").glob("*.log")) if (self.results_dir / "logs").exists() else []
            
            Logger.log("Results directory exists:")
            print(f"  - Result files: {len(result_files)}")
            print(f"  - Log files: {len(log_files)}")
            
            summary_file = self.results_dir / "summary.txt"
            if summary_file.exists():
                print("\nLatest summary:")
                with open(summary_file) as f:
                    lines = f.readlines()[:20]
                    print(''.join(lines))
        else:
            Logger.log("No results directory found")
    
    def cmd_stop(self):
        """Stop vLLM server"""
        Logger.log("Stopping vLLM server...")
        
        if not self.is_container_running():
            Logger.error(f"Container {self.container_name} is not running")
            return 1
        
        if self.is_vllm_running():
            try:
                self.docker_exec("pkill -f 'vllm serve'")
                import time
                time.sleep(2)
                
                if self.is_vllm_running():
                    Logger.warning("Forcing kill...")
                    self.docker_exec("pkill -9 -f 'vllm serve'")
                
                Logger.success("vLLM server stopped")
            except Exception as e:
                Logger.error(f"Failed to stop vLLM: {e}")
                return 1
        else:
            Logger.log("No vLLM server was running")
        
        return 0
    
    def cmd_clean(self):
        """Clean up results"""
        if not self.results_dir.exists():
            Logger.log("No results directory to clean")
            return 0
        
        result_files = list(self.results_dir.glob("*_results.json"))
        
        Logger.warning(f"This will delete {len(result_files)} result files and all logs")
        response = input("Are you sure? (yes/no): ")
        
        if response.lower() == 'yes':
            shutil.rmtree(self.results_dir)
            Logger.success("Results directory cleaned")
        else:
            Logger.log("Cleanup cancelled")
        
        return 0
    
    def cmd_backup(self):
        """Backup results"""
        if not self.results_dir.exists():
            Logger.error("No results directory to backup")
            return 1
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = Path(f"{self.results_dir}_{timestamp}")
        
        Logger.log(f"Creating backup: {backup_dir}")
        shutil.copytree(self.results_dir, backup_dir)
        
        Logger.success(f"Backup created: {backup_dir}")
        
        # Show backup size
        size = subprocess.run(
            ["du", "-sh", str(backup_dir)],
            capture_output=True,
            text=True
        ).stdout.split()[0]
        Logger.log(f"Backup size: {size}")
        
        return 0
    
    def cmd_check(self):
        """Check prerequisites"""
        Logger.log("Checking prerequisites...")
        print()
        
        all_good = True
        
        # Check Docker
        try:
            subprocess.run(["docker", "--version"], capture_output=True, timeout=5)
            Logger.success("Docker is installed")
        except Exception:
            Logger.error("Docker is not installed")
            all_good = False
        
        # Check container
        if self.is_container_running():
            Logger.success(f"Container {self.container_name} is running")
        else:
            result = self.run_cmd(["docker", "ps", "-a", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"])
            if self.container_name in result.stdout:
                Logger.warning(f"Container {self.container_name} exists but is not running")
                print(f"  Start it with: docker start {self.container_name}")
            else:
                Logger.error(f"Container {self.container_name} does not exist")
                print("  Create it using the commands in official.sh or EXPERIMENT_AUTOMATION.md")
                all_good = False
        
        # Check Python3
        try:
            subprocess.run(["python3", "--version"], capture_output=True, timeout=5)
            Logger.success("Python3 is installed")
        except Exception:
            Logger.error("Python3 is not installed")
            all_good = False
        
        # Check jq (optional)
        try:
            subprocess.run(["jq", "--version"], capture_output=True, timeout=5)
            Logger.success("jq is installed (for JSON parsing)")
        except Exception:
            Logger.warning("jq not found (optional, used for inline metrics)")
        
        # Check scripts
        if Path("./run_experiments.py").exists() and Path("./run_experiments.py").stat().st_mode & 0o111:
            Logger.success("run_experiments.py is executable")
        else:
            Logger.error("run_experiments.py not found or not executable")
            all_good = False
        
        if Path("./analyze_results.py").exists() and Path("./analyze_results.py").stat().st_mode & 0o111:
            Logger.success("analyze_results.py is executable")
        else:
            Logger.warning("analyze_results.py not found or not executable")
        
        print()
        if all_good:
            Logger.success("All critical prerequisites met! Ready to run experiments.")
            return 0
        else:
            Logger.error("Some prerequisites are missing. See above for details.")
            return 1
    
    def cmd_logs(self):
        """Tail latest logs"""
        log_dir = self.results_dir / "logs"
        if not log_dir.exists():
            Logger.error("No logs directory found")
            return 1
        
        log_files = sorted(log_dir.glob("*_server.log"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not log_files:
            Logger.error("No log files found")
            return 1
        
        latest_log = log_files[0]
        Logger.log(f"Tailing latest log: {latest_log}")
        print("Press Ctrl+C to stop")
        print()
        
        try:
            subprocess.run(["tail", "-f", str(latest_log)])
        except KeyboardInterrupt:
            print()
            Logger.log("Stopped tailing logs")
        
        return 0


def main():
    parser = argparse.ArgumentParser(
        description='vLLM Experiment Utilities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  status    Show current experiment status
  stop      Stop any running vLLM servers
  clean     Clean up result files (with confirmation)
  backup    Backup current results with timestamp
  check     Check prerequisites and container status
  logs      Tail the latest experiment logs

Examples:
  %(prog)s status       # Check what's currently running
  %(prog)s stop         # Stop vLLM server
  %(prog)s backup       # Backup results before new run
  %(prog)s clean        # Remove all results (prompts for confirmation)
        """
    )
    
    parser.add_argument(
        'command',
        choices=['status', 'stop', 'clean', 'backup', 'check', 'logs'],
        help='Command to execute'
    )
    parser.add_argument(
        '--container',
        default='vllm-test',
        help='Docker container name'
    )
    parser.add_argument(
        '--results-dir',
        default='./experiment_results',
        help='Results directory'
    )
    
    args = parser.parse_args()
    
    utils = ExperimentUtils(
        container_name=args.container,
        results_dir=args.results_dir
    )
    
    commands = {
        'status': utils.cmd_status,
        'stop': utils.cmd_stop,
        'clean': utils.cmd_clean,
        'backup': utils.cmd_backup,
        'check': utils.cmd_check,
        'logs': utils.cmd_logs
    }
    
    return commands[args.command]()


if __name__ == '__main__':
    sys.exit(main())
