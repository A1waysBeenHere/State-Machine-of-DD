#!/usr/bin/env python3
"""
Core analysis script for distributed training parallel strategy.
Calculates memory requirements, communication overhead, and recommends optimal configuration.
"""

import json
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from itertools import product


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    name: str
    params: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    intermediate_size: int
    vocab_size: int = 32000
    num_experts: int = 1
    experts_per_token: int = 1


@dataclass
class TrainingConfig:
    """Training configuration."""
    task: str  # "SFT", "pretrain", etc.
    seq_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    precision: str  # "fp32", "fp16", "bf16"
    activation_checkpointing: str  # "none", "selective", "full"


@dataclass
class HardwareConfig:
    """Hardware configuration."""
    gpu_type: str
    gpu_memory_gb: float
    num_gpus: int
    gpus_per_node: int
    nvlink_bandwidth_gbps: float
    internode_bandwidth_gbps: float
    gpu_peak_flops: float  # TFLOPS


@dataclass
class ParallelConfig:
    """Parallelism configuration."""
    dp_size: int
    tp_size: int
    pp_size: int
    sp_size: int
    ep_size: int = 1
    
    def total_gpus(self) -> int:
        return self.dp_size * self.tp_size * self.pp_size * self.sp_size


class ParallelStrategyAnalyzer:
    """Analyzer for distributed training parallel strategies."""
    
    # GPU specifications database
    GPU_SPECS = {
        "A100-40GB": {"memory": 40, "flops": 312, "nvlink": 600},
        "A100-80GB": {"memory": 80, "flops": 312, "nvlink": 600},
        "H100-80GB": {"memory": 80, "flops": 989, "nvlink": 900},
        "H100-NVL": {"memory": 188, "flops": 989, "nvlink": 900},
        "A10-24GB": {"memory": 24, "flops": 125, "nvlink": 0},
        "L40S-48GB": {"memory": 48, "flops": 183, "nvlink": 0},
        "MI250X": {"memory": 128, "flops": 383, "nvlink": 800},
        "MI300X": {"memory": 192, "flops": 1300, "nvlink": 896},
    }
    
    def __init__(
        self,
        model: ModelConfig,
        training: TrainingConfig,
        hardware: HardwareConfig,
        profiling_data: Optional[Dict] = None
    ):
        self.model = model
        self.training = training
        self.hardware = hardware
        self.profiling_data = profiling_data
        
    def bytes_per_param(self) -> int:
        """Get bytes per parameter based on precision."""
        return {"fp32": 4, "fp16": 2, "bf16": 2}.get(self.training.precision, 2)
    
    def calculate_memory(self, config: ParallelConfig) -> Dict[str, float]:
        """Calculate memory requirements per GPU in GB."""
        bytes_per_param = self.bytes_per_param()
        
        # Model parameters memory
        param_memory = self.model.params * bytes_per_param
        
        # Optimizer states (Adam: momentum + variance, both fp32)
        if config.dp_size > 0:
            optimizer_memory = self.model.params * 2 * 4  # fp32
        else:
            optimizer_memory = 0
        
        # Gradients
        grad_memory = self.model.params * bytes_per_param
        
        # Sharded model states (divided by DP size)
        sharded_model_states = (param_memory + optimizer_memory + grad_memory) / config.dp_size
        
        # Activation memory
        activation_memory = self._calculate_activation_memory(config)
        
        # Communication buffers
        comm_memory = self._calculate_communication_memory(config)
        
        # Convert to GB
        return {
            "model": sharded_model_states / 1e9,
            "optimizer": (optimizer_memory / config.dp_size) / 1e9,
            "gradient": (grad_memory / config.dp_size) / 1e9,
            "activation": activation_memory / 1e9,
            "communication": comm_memory / 1e9,
            "total": (sharded_model_states + activation_memory + comm_memory) / 1e9
        }
    
    def _calculate_activation_memory(self, config: ParallelConfig) -> float:
        """Calculate activation memory in bytes."""
        b = self.training.micro_batch_size
        s = self.training.seq_length
        h = self.model.hidden_size
        l = self.model.num_layers
        
        bytes_per_act = self.bytes_per_param()
        
        # Per layer activations (simplified)
        # Input + QKV + Attention Out + MLP Intermediate + MLP Out
        acts_per_layer = (
            b * s * h * 2 +      # Input
            b * s * h * 3 +      # Q, K, V
            b * s * h +          # Attention output
            b * s * self.model.intermediate_size +  # MLP intermediate
            b * s * h            # MLP output
        )
        
        # Apply TP/SP reduction
        acts_per_layer /= (config.tp_size * config.sp_size)
        
        # Apply checkpointing
        checkpointing = self.training.activation_checkpointing
        if checkpointing == "full":
            total_acts = b * s * h * l  # Only store layer inputs
        elif checkpointing == "selective":
            total_acts = acts_per_layer * l * 0.33
        else:
            total_acts = acts_per_layer * l
        
        return total_acts * bytes_per_act
    
    def _calculate_communication_memory(self, config: ParallelConfig) -> float:
        """Calculate communication buffer memory in bytes."""
        b = self.training.micro_batch_size
        s = self.training.seq_length / config.sp_size  # SP reduces seq per GPU
        h = self.model.hidden_size
        bytes_per_param = self.bytes_per_param()
        
        memory = 0
        
        # TP buffers
        if config.tp_size > 1:
            memory += 2 * b * s * h * bytes_per_param
        
        # PP buffers
        if config.pp_size > 1:
            memory += 2 * b * s * h * bytes_per_param
        
        return memory
    
    def calculate_communication_time(self, config: ParallelConfig) -> Dict[str, float]:
        """Calculate communication time in milliseconds."""
        bytes_per_param = self.bytes_per_param()
        
        results = {}
        
        # DP all-reduce time
        if config.dp_size > 1:
            grad_size = self.model.params * bytes_per_param / config.tp_size / config.pp_size
            comm_volume = 2 * (config.dp_size - 1) / config.dp_size * grad_size
            bandwidth = self.hardware.internode_bandwidth_gbps * 1e9 / 8
            results["dp_time_ms"] = (comm_volume / bandwidth) * 1000
        else:
            results["dp_time_ms"] = 0
        
        # TP all-reduce time
        if config.tp_size > 1:
            b = self.training.micro_batch_size
            s = self.training.seq_length / config.sp_size
            h = self.model.hidden_size
            
            act_size = b * s * h * bytes_per_param
            comm_per_layer = 2 * (config.tp_size - 1) / config.tp_size * act_size
            total_comm = comm_per_layer * 2 * self.model.num_layers  # 2 per layer
            
            bandwidth = self.hardware.nvlink_bandwidth_gbps * 1e9 / 8
            results["tp_time_ms"] = (total_comm / bandwidth) * 1000
        else:
            results["tp_time_ms"] = 0
        
        # PP P2P time
        if config.pp_size > 1:
            b = self.training.micro_batch_size
            s = self.training.seq_length / config.sp_size
            h = self.model.hidden_size
            
            act_size = b * s * h * bytes_per_param
            comm_volume = 2 * act_size  # Forward + backward
            
            bandwidth = self.hardware.nvlink_bandwidth_gbps * 1e9 / 8
            results["pp_time_ms"] = (comm_volume / bandwidth) * 1000
        else:
            results["pp_time_ms"] = 0
        
        # SP communication time
        if config.sp_size > 1:
            b = self.training.micro_batch_size
            s = self.training.seq_length
            h = self.model.hidden_size
            
            qkv_size = 3 * b * s * h * bytes_per_param
            output_size = b * s * h * bytes_per_param
            comm_per_layer = (config.sp_size - 1) / config.sp_size * (qkv_size + output_size)
            total_comm = comm_per_layer * self.model.num_layers
            
            bandwidth = self.hardware.nvlink_bandwidth_gbps * 1e9 / 8
            results["sp_time_ms"] = (total_comm / bandwidth) * 1000
        else:
            results["sp_time_ms"] = 0
        
        results["total_time_ms"] = sum(results.values())
        return results
    
    def calculate_compute_time(self, config: ParallelConfig) -> Dict[str, float]:
        """Calculate compute time estimates."""
        b = self.training.micro_batch_size * config.dp_size  # Global batch
        s = self.training.seq_length
        
        # Forward pass FLOPs
        forward_flops = 2 * self.model.params * b * s
        
        # Backward pass (~2x forward)
        total_flops = 3 * forward_flops
        
        # Per-GPU FLOPs
        flops_per_gpu = total_flops / config.dp_size / config.tp_size
        
        # Estimate MFU (typically 40-60% for well-optimized training)
        estimated_mfu = 0.5
        
        # Adjust MFU based on parallelism
        if config.tp_size > 4:
            estimated_mfu *= 0.9  # TP overhead
        if config.pp_size > 1:
            estimated_mfu *= 0.95  # PP bubble
        
        effective_flops = self.hardware.gpu_peak_flops * 1e12 * estimated_mfu
        compute_time_sec = flops_per_gpu / effective_flops
        
        return {
            "forward_time_ms": compute_time_sec * 1000 / 3,
            "backward_time_ms": compute_time_sec * 1000 * 2 / 3,
            "total_time_ms": compute_time_sec * 1000,
            "estimated_mfu": estimated_mfu
        }
    
    def calculate_throughput(self, config: ParallelConfig) -> float:
        """Calculate throughput in tokens/sec."""
        compute = self.calculate_compute_time(config)
        comm = self.calculate_communication_time(config)
        
        total_time_sec = (compute["total_time_ms"] + comm["total_time_ms"]) / 1000
        
        tokens_per_iter = (
            self.training.micro_batch_size * 
            config.dp_size * 
            self.training.seq_length
        )
        
        return tokens_per_iter / total_time_sec
    
    def calculate_mfu(self, config: ParallelConfig, throughput: float) -> float:
        """Calculate Model FLOPs Utilization."""
        # Peak FLOPs for all GPUs
        peak_flops = self.hardware.gpu_peak_flops * 1e12 * self.hardware.num_gpus
        
        # Actual FLOPs based on throughput
        # Each token requires ~6 * params FLOPs (2 forward + 4 backward)
        actual_flops = throughput * 6 * self.model.params
        
        return actual_flops / peak_flops
    
    def analyze_profiling_data(self) -> Dict:
        """Analyze profiling data for bottlenecks."""
        if not self.profiling_data:
            return {}
        
        insights = {
            "bottleneck_type": None,
            "recommendations": [],
            "metrics": {}
        }
        
        # Memory analysis
        if "memory_stats" in self.profiling_data:
            mem = self.profiling_data["memory_stats"]
            peak_pct = mem.get("peak_allocated", 0) / (self.hardware.gpu_memory_gb * 1e9)
            insights["metrics"]["memory_utilization"] = peak_pct
            
            if peak_pct > 0.95:
                insights["bottleneck_type"] = "memory_bound"
                insights["recommendations"].append("Increase TP/SP size to reduce activation memory")
                insights["recommendations"].append("Enable full activation checkpointing")
            elif peak_pct > 0.85:
                insights["recommendations"].append("Memory headroom is limited, monitor closely")
        
        # Communication analysis
        if "communication_stats" in self.profiling_data:
            comm = self.profiling_data["communication_stats"]
            comm_time = comm.get("all_reduce_total_time_ms", 0)
            total_time = comm_time + comm.get("compute_time_ms", 1)
            comm_ratio = comm_time / total_time
            
            insights["metrics"]["comm_ratio"] = comm_ratio
            
            if comm_ratio > 0.3:
                insights["bottleneck_type"] = "communication_bound"
                insights["recommendations"].append("Reduce TP size to decrease all-reduce overhead")
                insights["recommendations"].append("Consider increasing DP size instead")
        
        # Compute analysis
        if "compute_stats" in self.profiling_data:
            compute = self.profiling_data["compute_stats"]
            mfu = compute.get("mfu", 0)
            insights["metrics"]["measured_mfu"] = mfu
            
            if mfu < 0.3 and not insights["bottleneck_type"]:
                insights["bottleneck_type"] = "compute_underutilized"
                insights["recommendations"].append("Increase batch size to improve GPU utilization")
        
        return insights
    
    def score_configuration(self, config: ParallelConfig, metrics: Dict) -> float:
        """Score a configuration (higher is better)."""
        # Weights for different factors
        weights = {
            "throughput": 0.4,
            "mfu": 0.3,
            "memory_headroom": 0.2,
            "comm_efficiency": 0.1
        }
        
        # Normalize metrics
        throughput_score = min(metrics["throughput"] / 10000, 1.0)
        mfu_score = metrics["mfu"]
        memory_headroom = 1.0 - (metrics["memory"]["total"] / self.hardware.gpu_memory_gb)
        memory_score = max(0, memory_headroom)
        comm_efficiency = 1.0 - min(metrics["comm_time_ms"] / 1000, 1.0)
        
        return (
            weights["throughput"] * throughput_score +
            weights["mfu"] * mfu_score +
            weights["memory_headroom"] * memory_score +
            weights["comm_efficiency"] * comm_efficiency
        )
    
    def generate_configurations(self) -> List[ParallelConfig]:
        """Generate valid parallel configurations."""
        configs = []
        total_gpus = self.hardware.num_gpus
        
        # Generate all valid combinations
        for tp in [1, 2, 4, 8]:
            for pp in [1, 2, 4, 8, 16]:
                for sp in [1, tp]:  # SP usually equals TP or 1
                    product = tp * pp * sp
                    if total_gpus % product != 0:
                        continue
                    dp = total_gpus // product
                    
                    # Constraints
                    if tp > 8:  # TP typically limited to 8
                        continue
                    if pp > self.model.num_layers:  # Can't have more stages than layers
                        continue
                    if self.model.num_experts > 1 and tp > 1:
                        # For MoE, EP usually replaces TP
                        continue
                    
                    configs.append(ParallelConfig(
                        dp_size=dp,
                        tp_size=tp,
                        pp_size=pp,
                        sp_size=sp,
                        ep_size=1 if self.model.num_experts == 1 else min(self.model.num_experts, dp)
                    ))
        
        return configs
    
    def analyze(self) -> Dict:
        """Run full analysis and return results."""
        configs = self.generate_configurations()
        results = []
        
        for config in configs:
            memory = self.calculate_memory(config)
            comm = self.calculate_communication_time(config)
            compute = self.calculate_compute_time(config)
            throughput = self.calculate_throughput(config)
            mfu = self.calculate_mfu(config, throughput)
            
            # Check validity
            valid = memory["total"] < self.hardware.gpu_memory_gb * 0.95
            
            metrics = {
                "memory": memory,
                "comm_time_ms": comm["total_time_ms"],
                "compute_time_ms": compute["total_time_ms"],
                "throughput": throughput,
                "mfu": mfu
            }
            
            score = self.score_configuration(config, metrics) if valid else 0
            
            results.append({
                "dp": config.dp_size,
                "tp": config.tp_size,
                "pp": config.pp_size,
                "sp": config.sp_size,
                "ep": config.ep_size,
                "memory": memory,
                "communication": comm,
                "compute": compute,
                "performance": {
                    "tokens_per_sec": throughput,
                    "mfu": mfu
                },
                "valid": valid,
                "score": score
            })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Profiling insights
        profiling_insights = self.analyze_profiling_data()
        
        return {
            "model": {
                "name": self.model.name,
                "params": self.model.params,
                "hidden_size": self.model.hidden_size,
                "num_layers": self.model.num_layers
            },
            "hardware": {
                "gpu_type": self.hardware.gpu_type,
                "num_gpus": self.hardware.num_gpus,
                "gpu_memory_gb": self.hardware.gpu_memory_gb
            },
            "training": {
                "task": self.training.task,
                "seq_length": self.training.seq_length,
                "batch_size": self.training.micro_batch_size
            },
            "configurations": results,
            "profiling_insights": profiling_insights,
            "recommendation": results[0] if results else None
        }


def load_config_from_json(path: str) -> Tuple[ModelConfig, TrainingConfig, HardwareConfig, Optional[Dict]]:
    """Load configuration from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    model = ModelConfig(**data["model"])
    training = TrainingConfig(**data["training"])
    hardware = HardwareConfig(**data["hardware"])
    profiling = data.get("profiling")
    
    return model, training, hardware, profiling


def main():
    parser = argparse.ArgumentParser(
        description='Analyze distributed training parallel strategy'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='analysis_results.json',
        help='Path to save analysis results'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    model, training, hardware, profiling = load_config_from_json(args.config)
    
    # Run analysis
    analyzer = ParallelStrategyAnalyzer(model, training, hardware, profiling)
    results = analyzer.analyze()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Analysis complete. Results saved to: {args.output}")
    
    # Print summary
    if results["recommendation"]:
        rec = results["recommendation"]
        print("\n=== Recommended Configuration ===")
        print(f"DP={rec['dp']}, TP={rec['tp']}, PP={rec['pp']}, SP={rec['sp']}")
        print(f"Expected Throughput: {rec['performance']['tokens_per_sec']:.0f} tokens/sec")
        print(f"Expected MFU: {rec['performance']['mfu']*100:.1f}%")
        print(f"Memory Utilization: {rec['memory']['total']:.1f} / {hardware.gpu_memory_gb} GB")


if __name__ == '__main__':
    main()
