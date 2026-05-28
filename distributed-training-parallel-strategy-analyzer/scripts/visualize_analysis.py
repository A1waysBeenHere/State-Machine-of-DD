#!/usr/bin/env python3
"""
Visualization script for parallel strategy analysis.
Generates charts for memory breakdown, communication overhead, and performance comparison.
"""

import json
import argparse
from typing import Dict, List, Any
import os

# Try to import matplotlib, provide fallback if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib numpy")


def generate_memory_breakdown_chart(configs: List[Dict], output_path: str):
    """Generate stacked bar chart showing memory breakdown per configuration."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping memory chart - matplotlib not available")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    labels = [f"DP={c['dp']},TP={c['tp']},PP={c['pp']}" for c in configs]
    
    # Memory components
    model_mem = [c['memory']['model'] for c in configs]
    optimizer_mem = [c['memory']['optimizer'] for c in configs]
    grad_mem = [c['memory']['gradient'] for c in configs]
    activation_mem = [c['memory']['activation'] for c in configs]
    comm_mem = [c['memory']['communication'] for c in configs]
    
    x = np.arange(len(labels))
    width = 0.6
    
    # Stacked bars
    bottom1 = np.array(model_mem)
    bottom2 = bottom1 + np.array(optimizer_mem)
    bottom3 = bottom2 + np.array(grad_mem)
    bottom4 = bottom3 + np.array(activation_mem)
    
    ax.bar(x, model_mem, width, label='Model Params', color='#3498db')
    ax.bar(x, optimizer_mem, width, bottom=bottom1, label='Optimizer States', color='#e74c3c')
    ax.bar(x, grad_mem, width, bottom=bottom2, label='Gradients', color='#f39c12')
    ax.bar(x, activation_mem, width, bottom=bottom3, label='Activations', color='#2ecc71')
    ax.bar(x, comm_mem, width, bottom=bottom4, label='Communication Buffers', color='#9b59b6')
    
    # GPU memory limit line
    gpu_memory = configs[0].get('gpu_memory_gb', 80)
    ax.axhline(y=gpu_memory * 0.95, color='red', linestyle='--', linewidth=2, label=f'GPU Limit (95%)')
    ax.axhline(y=gpu_memory * 0.85, color='orange', linestyle='--', linewidth=1, label=f'Warning (85%)')
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Memory (GB)')
    ax.set_title('Memory Breakdown by Parallel Configuration')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Memory breakdown chart saved to: {output_path}")


def generate_communication_overhead_chart(configs: List[Dict], output_path: str):
    """Generate chart showing communication overhead breakdown."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping communication chart - matplotlib not available")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    labels = [f"DP={c['dp']},TP={c['tp']},PP={c['pp']}" for c in configs]
    
    # Communication times
    dp_comm = [c['communication'].get('dp_time_ms', 0) for c in configs]
    tp_comm = [c['communication'].get('tp_time_ms', 0) for c in configs]
    pp_comm = [c['communication'].get('pp_time_ms', 0) for c in configs]
    sp_comm = [c['communication'].get('sp_time_ms', 0) for c in configs]
    
    x = np.arange(len(labels))
    width = 0.6
    
    bottom1 = np.array(dp_comm)
    bottom2 = bottom1 + np.array(tp_comm)
    bottom3 = bottom2 + np.array(pp_comm)
    
    ax.bar(x, dp_comm, width, label='Data Parallel', color='#3498db')
    ax.bar(x, tp_comm, width, bottom=bottom1, label='Tensor Parallel', color='#e74c3c')
    ax.bar(x, pp_comm, width, bottom=bottom2, label='Pipeline Parallel', color='#f39c12')
    ax.bar(x, sp_comm, width, bottom=bottom3, label='Sequence Parallel', color='#2ecc71')
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Communication Time (ms)')
    ax.set_title('Communication Overhead by Parallel Configuration')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Communication overhead chart saved to: {output_path}")


def generate_performance_comparison_chart(configs: List[Dict], output_path: str):
    """Generate chart comparing throughput and MFU across configurations."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping performance chart - matplotlib not available")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    labels = [f"DP={c['dp']},TP={c['tp']},PP={c['pp']}" for c in configs]
    x = np.arange(len(labels))
    
    # Throughput
    throughputs = [c['performance'].get('tokens_per_sec', 0) for c in configs]
    colors = ['#2ecc71' if c['valid'] else '#e74c3c' for c in configs]
    
    bars1 = ax1.bar(x, throughputs, color=colors, alpha=0.8)
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Throughput (tokens/sec)')
    ax1.set_title('Throughput Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, throughputs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}',
                ha='center', va='bottom', fontsize=8)
    
    # MFU
    mfus = [c['performance'].get('mfu', 0) * 100 for c in configs]
    bars2 = ax2.bar(x, mfus, color=colors, alpha=0.8)
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('MFU (%)')
    ax2.set_title('Model FLOPs Utilization')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars2, mfus):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=8)
    
    # Legend
    valid_patch = mpatches.Patch(color='#2ecc71', label='Valid')
    invalid_patch = mpatches.Patch(color='#e74c3c', label='OOM Risk')
    fig.legend(handles=[valid_patch, invalid_patch], loc='upper center', ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Performance comparison chart saved to: {output_path}")


def generate_pipeline_bubble_chart(config: Dict, output_path: str):
    """Generate visualization of pipeline bubble overhead."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping pipeline bubble chart - matplotlib not available")
        return
    
    if config.get('pp', 1) == 1:
        print("No pipeline parallelism - skipping bubble chart")
        return
    
    pp_size = config['pp']
    num_microbatches = config.get('num_microbatches', pp_size * 2)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Timeline visualization
    colors = {'forward': '#3498db', 'backward': '#e74c3c', 'bubble': '#95a5a6'}
    
    stage_height = 1.0
    y_positions = {i: i * stage_height for i in range(pp_size)}
    
    # Simplified pipeline schedule visualization
    # Warmup phase
    time = 0
    forward_time = config.get('forward_time', 100)
    backward_time = config.get('backward_time', 200)
    
    for stage in range(pp_size):
        for mb in range(num_microbatches):
            # Forward
            start = time + stage * forward_time + mb * forward_time
            ax.barh(y_positions[stage], forward_time, left=start, 
                   height=0.8, color=colors['forward'], alpha=0.8)
            
            # Backward
            start_bwd = start + (pp_size - stage) * forward_time + backward_time
            ax.barh(y_positions[stage], backward_time, left=start_bwd,
                   height=0.8, color=colors['backward'], alpha=0.8)
    
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([f'Stage {i}' for i in range(pp_size)])
    ax.set_xlabel('Time')
    ax.set_title('Pipeline Parallel Schedule (Simplified)')
    ax.grid(axis='x', alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['forward'], label='Forward'),
        Patch(facecolor=colors['backward'], label='Backward'),
        Patch(facecolor=colors['bubble'], label='Bubble')
    ]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Pipeline bubble chart saved to: {output_path}")


def generate_profiling_heatmap(profiling_data: Dict, output_path: str):
    """Generate heatmap from profiling data showing hotspots."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping profiling heatmap - matplotlib not available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Memory timeline
    if 'memory_timeline' in profiling_data:
        ax = axes[0, 0]
        timeline = profiling_data['memory_timeline']
        times = [t['time_ms'] for t in timeline]
        allocated = [t['allocated_gb'] for t in timeline]
        reserved = [t['reserved_gb'] for t in timeline]
        
        ax.plot(times, allocated, label='Allocated', color='#3498db')
        ax.plot(times, reserved, label='Reserved', color='#e74c3c')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Memory (GB)')
        ax.set_title('Memory Timeline')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Communication breakdown
    if 'communication' in profiling_data:
        ax = axes[0, 1]
        comm = profiling_data['communication']
        labels = list(comm.keys())
        values = list(comm.values())
        
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title('Communication Breakdown')
    
    # Layer-wise time
    if 'layer_times' in profiling_data:
        ax = axes[1, 0]
        layers = profiling_data['layer_times']
        layer_ids = list(range(len(layers)))
        forward_times = [l['forward_ms'] for l in layers]
        backward_times = [l['backward_ms'] for l in layers]
        
        x = np.arange(len(layer_ids))
        width = 0.35
        ax.bar(x - width/2, forward_times, width, label='Forward', color='#3498db')
        ax.bar(x + width/2, backward_times, width, label='Backward', color='#e74c3c')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Layer-wise Execution Time')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    # GPU utilization
    if 'gpu_utilization' in profiling_data:
        ax = axes[1, 1]
        util = profiling_data['gpu_utilization']
        times = [u['time_ms'] for u in util]
        compute = [u['compute_pct'] for u in util]
        memory = [u['memory_pct'] for u in util]
        
        ax.plot(times, compute, label='Compute', color='#2ecc71')
        ax.plot(times, memory, label='Memory', color='#f39c12')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Utilization (%)')
        ax.set_title('GPU Utilization')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Profiling heatmap saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate visualization charts for parallel strategy analysis'
    )
    parser.add_argument(
        '--analysis-json',
        type=str,
        required=True,
        help='Path to analysis results JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./visualizations',
        help='Directory to save output charts'
    )
    parser.add_argument(
        '--profiling-json',
        type=str,
        help='Optional path to profiling data JSON'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load analysis results
    with open(args.analysis_json, 'r') as f:
        analysis_data = json.load(f)
    
    configs = analysis_data.get('configurations', [])
    
    if not configs:
        print("No configurations found in analysis data")
        return
    
    # Generate charts
    base_name = os.path.splitext(os.path.basename(args.analysis_json))[0]
    
    generate_memory_breakdown_chart(
        configs,
        os.path.join(args.output_dir, f'{base_name}_memory.png')
    )
    
    generate_communication_overhead_chart(
        configs,
        os.path.join(args.output_dir, f'{base_name}_communication.png')
    )
    
    generate_performance_comparison_chart(
        configs,
        os.path.join(args.output_dir, f'{base_name}_performance.png')
    )
    
    # Pipeline bubble for best config with PP
    best_config = max(configs, key=lambda x: x.get('score', 0))
    if best_config.get('pp', 1) > 1:
        generate_pipeline_bubble_chart(
            best_config,
            os.path.join(args.output_dir, f'{base_name}_pipeline.png')
        )
    
    # Profiling heatmap if data provided
    if args.profiling_json:
        with open(args.profiling_json, 'r') as f:
            profiling_data = json.load(f)
        generate_profiling_heatmap(
            profiling_data,
            os.path.join(args.output_dir, f'{base_name}_profiling.png')
        )
    
    print(f"\nAll visualizations saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
