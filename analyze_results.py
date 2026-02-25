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


def analyze_results(results_dir: str = './experiment_results') -> None:
    """Analyze all result files and generate comprehensive report"""
    
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: Results directory '{results_dir}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Find all result JSON files
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
            results.append({
                'config': config,
                'data': data,
                'filename': file_path.name
            })
    
    # Generate analysis report
    print("=" * 80)
    print("vLLM Benchmark Results Analysis")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total successful experiments: {len(results)}\n")
    
    # Overall statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    
    all_throughputs = []
    all_output_throughputs = []
    all_ttfts = []
    all_tpots = []
    
    for result in results:
        data = result['data']
        
        req_throughput = data.get('request_throughput')
        out_throughput = data.get('output_throughput')
        ttft = data.get('mean_ttft_ms')
        tpot = data.get('mean_tpot_ms')
        
        if req_throughput is not None:
            all_throughputs.append(float(req_throughput))
        if out_throughput is not None:
            all_output_throughputs.append(float(out_throughput))
        if ttft is not None:
            all_ttfts.append(float(ttft))
        if tpot is not None:
            all_tpots.append(float(tpot))
    
    if all_throughputs:
        print(f"Request throughput (req/s):")
        print(f"  Min:    {min(all_throughputs):.2f}")
        print(f"  Max:    {max(all_throughputs):.2f}")
        print(f"  Mean:   {sum(all_throughputs)/len(all_throughputs):.2f}")
        print(f"  Median: {sorted(all_throughputs)[len(all_throughputs)//2]:.2f}")
    
    if all_output_throughputs:
        print(f"\nOutput throughput (tok/s):")
        print(f"  Min:    {min(all_output_throughputs):.2f}")
        print(f"  Max:    {max(all_output_throughputs):.2f}")
        print(f"  Mean:   {sum(all_output_throughputs)/len(all_output_throughputs):.2f}")
        print(f"  Median: {sorted(all_output_throughputs)[len(all_output_throughputs)//2]:.2f}")
    
    if all_ttfts:
        print(f"\nMean TTFT (ms):")
        print(f"  Min:    {min(all_ttfts):.2f}")
        print(f"  Max:    {max(all_ttfts):.2f}")
        print(f"  Mean:   {sum(all_ttfts)/len(all_ttfts):.2f}")
        print(f"  Median: {sorted(all_ttfts)[len(all_ttfts)//2]:.2f}")
    
    if all_tpots:
        print(f"\nMean TPOT (ms):")
        print(f"  Min:    {min(all_tpots):.2f}")
        print(f"  Max:    {max(all_tpots):.2f}")
        print(f"  Mean:   {sum(all_tpots)/len(all_tpots):.2f}")
        print(f"  Median: {sorted(all_tpots)[len(all_tpots)//2]:.2f}")
    
    # Group by model
    print("\n" + "=" * 80)
    print("RESULTS BY MODEL")
    print("=" * 80)
    
    models = {}
    for result in results:
        model = result['config']['model']
        if model not in models:
            models[model] = []
        models[model].append(result)
    
    for model, model_results in sorted(models.items()):
        print(f"\n{model} ({len(model_results)} configurations)")
        print("-" * 80)
        
        # Create a table
        print(f"{'TP':<4} {'Quant':<6} {'Eager':<6} {'Req/s':<10} {'Out tok/s':<12} {'TTFT (ms)':<12} {'TPOT (ms)':<12} {'Status':<10}")
        print("-" * 80)
        
        for result in sorted(model_results, key=lambda x: (x['config']['tp'], x['config']['quant'], x['config']['eager'])):
            config = result['config']
            data = result['data']
            
            req_tput = data.get('request_throughput')
            out_tput = data.get('output_throughput')
            ttft = data.get('mean_ttft_ms')
            tpot = data.get('mean_tpot_ms')
            
            req_tput_str = f"{req_tput:.2f}" if req_tput is not None else 'N/A'
            out_tput_str = f"{out_tput:.2f}" if out_tput is not None else 'N/A'
            ttft_str = f"{ttft:.2f}" if ttft is not None else 'N/A'
            tpot_str = f"{tpot:.2f}" if tpot is not None else 'N/A'
            status = "✓ Success"
            
            print(f"{config['tp']:<4} {config['quant']:<6} {config['eager']:<6} {req_tput_str:<10} {out_tput_str:<12} {ttft_str:<12} {tpot_str:<12} {status:<10}")
    
    # Best configurations
    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS")
    print("=" * 80)
    
    if all_throughputs:
        # Find best throughput
        best_throughput_idx = all_throughputs.index(max(all_throughputs))
        best_result = results[best_throughput_idx]
        print(f"\nHighest Throughput: {max(all_throughputs):.2f} tokens/s")
        print(f"  Configuration: {best_result['filename']}")
        print(f"  Model: {best_result['config']['model']}")
        print(f"  TP: {best_result['config']['tp']}")
        print(f"  Quantization: {best_result['config']['quant']}")
        print(f"  Enforce Eager: {best_result['config']['eager']}")
    
    if all_ttfts:
        # Find best TTFT (lowest)
        best_ttft_idx = all_ttfts.index(min(all_ttfts))
        best_result = results[best_ttft_idx]
        print(f"\nLowest TTFT: {min(all_ttfts):.2f} ms")
        print(f"  Configuration: {best_result['filename']}")
        print(f"  Model: {best_result['config']['model']}")
        print(f"  TP: {best_result['config']['tp']}")
        print(f"  Quantization: {best_result['config']['quant']}")
        print(f"  Enforce Eager: {best_result['config']['eager']}")
    
    if all_tpots:
        # Find best TPOT (lowest)
        best_tpot_idx = all_tpots.index(min(all_tpots))
        best_result = results[best_tpot_idx]
        print(f"\nLowest TPOT: {min(all_tpots):.2f} ms")
        print(f"  Configuration: {best_result['filename']}")
        print(f"  Model: {best_result['config']['model']}")
        print(f"  TP: {best_result['config']['tp']}")
        print(f"  Quantization: {best_result['config']['quant']}")
        print(f"  Enforce Eager: {best_result['config']['eager']}")
    
    # Save detailed report
    report_file = results_path / 'detailed_analysis.txt'
    with open(report_file, 'w') as f:
        f.write(f"Detailed Analysis Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for result in results:
            f.write("=" * 80 + "\n")
            f.write(f"Configuration: {result['filename']}\n")
            f.write("-" * 80 + "\n")
            f.write(json.dumps(result['data'], indent=2))
            f.write("\n\n")
    
    print(f"\n\nDetailed report saved to: {report_file}")
    
    # Generate CSV for easy import to spreadsheets
    csv_file = results_path / 'results_summary.csv'
    with open(csv_file, 'w') as f:
        f.write("Model,TP,Quantization,EnforceEager,ReqThroughput_req_s,OutThroughput_tok_s,TTFT_ms,TPOT_ms,Filename\n")
        for result in results:
            config = result['config']
            data = result['data']
            
            req_tput = data.get('request_throughput', '')
            out_tput = data.get('output_throughput', '')
            ttft = data.get('mean_ttft_ms', '')
            tpot = data.get('mean_tpot_ms', '')
            
            f.write(f"{config['model']},{config['tp']},{config['quant']},{config['eager']},{req_tput},{out_tput},{ttft},{tpot},{result['filename']}\n")
    
    print(f"CSV summary saved to: {csv_file}")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze vLLM benchmark results')
    parser.add_argument('--results-dir', default='./experiment_results',
                       help='Directory containing result files (default: ./experiment_results)')
    
    args = parser.parse_args()
    analyze_results(args.results_dir)
