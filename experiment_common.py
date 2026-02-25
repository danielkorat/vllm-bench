#!/usr/bin/env python3
"""
Shared data classes and utilities for vLLM experiment scripts.
Used by run_experiments.py and other scripts.
"""

from datetime import datetime
from typing import Optional
from dataclasses import dataclass


class Color:
    """ANSI color codes for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    model: str
    tp: int
    quantization: Optional[str]  # None means no quantization
    enforce_eager: bool

    @property
    def name(self) -> str:
        """Generate a unique name for this configuration"""
        model_sanitized = self.model.replace('/', '_')
        quant_str = self.quantization if self.quantization else 'none'
        eager_str = 'true' if self.enforce_eager else 'false'
        return f"{model_sanitized}_tp{self.tp}_quant-{quant_str}_eager-{eager_str}"


@dataclass
class ExperimentResult:
    """Result of running a single experiment"""
    config: ExperimentConfig
    success: bool
    error_message: Optional[str] = None
    duration: Optional[float] = None


class Logger:
    """Colored logging utility"""

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
