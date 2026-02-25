#!/usr/bin/env python3
"""
Analyze and visualize vLLM benchmark experiment results
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


def load_result(file_path: Path) -> Dict[str, Any]:
    """Load a single result file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}", file=sys.stderr)
        return {}


def parse_config_from_filename(filename: str) -> Dict[str, str]:
    """Parse configuration from filename"""
    # Format: model_tp#_quant-X_eager-Y_results.json
    parts = filename.replace('_results.json', '').split('_')
    
    config = {}
    model_parts = []
    
    for part in parts:
        if part.startswith('tp'):
            config['tp'] = part[2:]
        elif part.startswith('quant-'):
            config['quant'] = part[6:]
        elif part.startswith('eager-'):
            config['eager'] = part[6:]
        else:
            model_parts.append(part)
    
    config['model'] = '_'.join(model_parts)
    return config


def _best_config_block(label: str, results: List[Dict], metric_key: str, higher_is_better: bool) -> List[str]:
    """Return formatted lines for the best result on a given metric."""
    candidates = [(r, r['data'].get(metric_key)) for r in results if r['data'].get(metric_key) is not None]
    if not candidates:
        return []
    best_r, best_val = (max if higher_is_better else min)(candidates, key=lambda x: float(x[1]))
    unit = "req/s" if "throughput" in metric_key else ("tok/s" if "output" in metric_key else "ms")
    direction = "Highest" if higher_is_better else "Lowest"
    c = best_r['config']
    return [
        f"{direction} {label}: {float(best_val):.2f} {unit}",
        f"  Model: {c['model']}  TP: {c['tp']}  Quant: {c['quant']}  Eager: {c['eager']}",
        f"  File:  {best_r['filename']}",
    ]


def _stats_block(label: str, values: List[float], unit: str) -> List[str]:
    if not values:
        return []
    sv = sorted(values)
    median = sv[len(sv) // 2]
    return [
        f"{label} ({unit}):",
        f"  Min:    {min(values):.2f}",
        f"  Max:    {max(values):.2f}",
        f"  Mean:   {sum(values)/len(values):.2f}",
        f"  Median: {median:.2f}",
    ]


def analyze_results(results_dir: str = './experiment_results') -> None:
    """Analyze all result files, print to stdout, and save to detailed_analysis.txt."""

    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: Results directory '{results_dir}' not found", file=sys.stderr)
        sys.exit(1)

    result_files = list(results_path.glob('*_results.json'))
    if not result_files:
        print(f"No result files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(result_files)} result files\n")

    # Collect all results
    results = []
    for file_path in sorted(result_files):
        data = load_result(file_path)
        if data:
            config = parse_config_from_filename(file_path.name)
            results.append({'config': config, 'data': data, 'filename': file_path.name})

    # ------------------------------------------------------------------ #
    # Build the full report as a list of lines so we can both print it    #
    # and save it to a file without duplication.                           #
    # ------------------------------------------------------------------ #
    SEP = "=" * 80
    sep = "-" * 80
    lines: List[str] = []

    def section(title: str):
        lines.extend(["", SEP, title, SEP])

    # Header
    lines += [
        SEP,
        "vLLM Benchmark Results Analysis",
        SEP,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total successful experiments: {len(results)}",
    ]

    # Overall statistics
    all_throughputs:        List[float] = []
    all_output_throughputs: List[float] = []
    all_ttfts:              List[float] = []
    all_tpots:              List[float] = []

    for r in results:
        d = r['data']
        if d.get('request_throughput') is not None:
            all_throughputs.append(float(d['request_throughput']))
        if d.get('output_throughput') is not None:
            all_output_throughputs.append(float(d['output_throughput']))
        if d.get('mean_ttft_ms') is not None:
            all_ttfts.append(float(d['mean_ttft_ms']))
        if d.get('mean_tpot_ms') is not None:
            all_tpots.append(float(d['mean_tpot_ms']))

    section("OVERALL STATISTICS")
    lines += _stats_block("Request throughput", all_throughputs, "req/s")
    lines += ([""] if all_throughputs and all_output_throughputs else [])
    lines += _stats_block("Output throughput",  all_output_throughputs, "tok/s")
    lines += ([""] if all_output_throughputs and all_ttfts else [])
    lines += _stats_block("Mean TTFT",           all_ttfts, "ms")
    lines += ([""] if all_ttfts and all_tpots else [])
    lines += _stats_block("Mean TPOT",           all_tpots, "ms")

    # Results by model
    section("RESULTS BY MODEL")

    models: Dict[str, List] = {}
    for r in results:
        models.setdefault(r['config']['model'], []).append(r)

    HDR = f"{'TP':<4} {'Quant':<6} {'Eager':<6} {'Req/s':<10} {'Out tok/s':<12} {'TTFT (ms)':<12} {'TPOT (ms)':<12} {'Status':<10}"

    for model, model_results in sorted(models.items()):
        lines += ["", f"{model} ({len(model_results)} configurations)", sep, HDR, sep]
        for r in sorted(model_results, key=lambda x: (x['config']['tp'], x['config']['quant'], x['config']['eager'])):
            c, d = r['config'], r['data']
            req  = d.get('request_throughput')
            out  = d.get('output_throughput')
            ttft = d.get('mean_ttft_ms')
            tpot = d.get('mean_tpot_ms')
            lines.append(
                f"{c['tp']:<4} {c['quant']:<6} {c['eager']:<6} "
                f"{(f'{req:.2f}' if req is not None else 'N/A'):<10} "
                f"{(f'{out:.2f}' if out is not None else 'N/A'):<12} "
                f"{(f'{ttft:.2f}' if ttft is not None else 'N/A'):<12} "
                f"{(f'{tpot:.2f}' if tpot is not None else 'N/A'):<12} "
                f"{'✓ Success':<10}"
            )

    # Best configurations per model
    section("BEST CONFIGURATIONS PER MODEL")
    for model, model_results in sorted(models.items()):
        lines += ["", f"── {model} ──"]
        lines += _best_config_block("Request Throughput", model_results, 'request_throughput', higher_is_better=True)
        lines.append("")
        lines += _best_config_block("TTFT",               model_results, 'mean_ttft_ms',        higher_is_better=False)
        lines.append("")
        lines += _best_config_block("TPOT",               model_results, 'mean_tpot_ms',         higher_is_better=False)

    # Best configurations overall
    section("BEST CONFIGURATIONS OVERALL")
    lines.append("")
    lines += _best_config_block("Request Throughput", results, 'request_throughput', higher_is_better=True)
    lines.append("")
    lines += _best_config_block("Output Throughput",  results, 'output_throughput',  higher_is_better=True)
    lines.append("")
    lines += _best_config_block("TTFT",               results, 'mean_ttft_ms',        higher_is_better=False)
    lines.append("")
    lines += _best_config_block("TPOT",               results, 'mean_tpot_ms',         higher_is_better=False)
    lines.append("")

    # Print to stdout
    report_text = "\n".join(lines)
    print(report_text)

    # Save human-readable report
    report_file = results_path / 'detailed_analysis.txt'
    report_file.write_text(report_text + "\n")
    print(f"\nAnalysis saved to: {report_file}")

    # Save raw JSON dump
    raw_file = results_path / 'raw_results.json'
    with open(raw_file, 'w') as f:
        json.dump([{'filename': r['filename'], 'config': r['config'], 'data': r['data']} for r in results], f, indent=2)
    print(f"Raw JSON dump saved to: {raw_file}")

    # Generate CSV
    csv_file = results_path / 'results_summary.csv'
    with open(csv_file, 'w') as f:
        f.write("Model,TP,Quantization,EnforceEager,ReqThroughput_req_s,OutThroughput_tok_s,TTFT_ms,TPOT_ms,Filename\n")
        for r in results:
            c, d = r['config'], r['data']
            f.write(
                f"{c['model']},{c['tp']},{c['quant']},{c['eager']},"
                f"{d.get('request_throughput','')},{d.get('output_throughput','')},"
                f"{d.get('mean_ttft_ms','')},{d.get('mean_tpot_ms','')},{r['filename']}\n"
            )
    print(f"CSV summary saved to:  {csv_file}")
    print("\n" + SEP)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze vLLM benchmark results')
    parser.add_argument('--results-dir', default='./experiment_results',
                       help='Directory containing result files (default: ./experiment_results)')
    
    args = parser.parse_args()
    analyze_results(args.results_dir)
