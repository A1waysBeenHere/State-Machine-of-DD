---
name: distributed-training-parallel-strategy-analyzer
description: Analyzes and recommends optimal distributed training parallel strategies for LLM training. Considers model parameters, sequence length, hardware specs, GPU count, and model architecture to recommend DP/TP/PP/SP/EP configurations. Use when user needs to optimize distributed training setup, analyze profiling data for bottlenecks, or find max throughput/MFU without OOM. Supports SFT pre-training scenarios.
---

# Distributed Training Parallel Strategy Analyzer

## Overview

This skill analyzes distributed training configurations for Large Language Models and recommends optimal parallel strategies to maximize throughput/MFU while avoiding OOM errors.

## Input Requirements

User must provide:

### 1. Model Configuration
- Model architecture (e.g., Llama, GPT, Qwen)
- Number of parameters (e.g., 7B, 13B, 70B)
- Hidden size, number of layers, attention heads
- Vocabulary size
- MLP intermediate size

### 2. Training Configuration
- Sequence length (SFT场景下需区分prompt和completion长度)
- Batch size (micro batch size per GPU)
- Gradient accumulation steps
- Training precision (fp16/bf16/fp32)
- Activation checkpointing strategy

### 3. Hardware Configuration
- GPU type (e.g., A100-80GB, H100-80GB, A10-24GB)
- Number of GPUs
- Node configuration (GPUs per node, node count)
- Inter-GPU bandwidth (NVLink/PCIe)
- Inter-node bandwidth (InfiniBand/Ethernet)

### 4. Profiling Data (Optional but Recommended)
- Memory profiling trace
- Communication profiling trace
- Compute profiling trace
- FLOPs utilization data

## Analysis Workflow

### Phase 1: Memory Analysis

Calculate memory requirements for each parallel strategy:

```python
# Memory components to calculate:
# 1. Model parameters memory
param_memory = params_count * precision_bytes

# 2. Optimizer states memory (Adam: 2x param memory for momentum + variance)
optimizer_memory = param_memory * 2  # for Adam

# 3. Gradients memory
grad_memory = param_memory

# 4. Activations memory (depends on TP/SP)
activation_memory = calculate_activations(seq_len, batch_size, hidden_size, layers, tp_size, sp_size)

# 5. Communication buffers
comm_memory = estimate_communication_buffers(tp_size, pp_size)

# Total per GPU
total_memory = (param_memory + optimizer_memory + grad_memory) / dp_size + activation_memory + comm_memory
```

**Memory constraints check:**
- If total_memory > GPU_memory * 0.95 → Strategy invalid (OOM risk)
- If total_memory > GPU_memory * 0.85 → Warning (limited headroom)

### Phase 2: Communication Analysis

Analyze communication overhead for each strategy:

| Parallel Type | Communication Pattern | Cost Factor |
|--------------|----------------------|-------------|
| DP | All-Reduce on gradients | 2*(N-1)/N * model_size |
| TP | All-Reduce in forward/backward | 2*seq_len*hidden_size*layers*(tp_size-1) |
| PP | P2P send/recv activations | 2*batch_size*seq_len*hidden_size*(pp_size-1) |
| SP | All-Gather/Reduce-Scatter | seq_len*hidden_size*sp_size |
| EP | All-to-All for MoE | 2*tokens_per_expert*hidden_size*ep_size |

**Communication bottleneck detection:**
- Compare communication time vs compute time
- If comm_time > 0.3 * compute_time → Communication bound
- Identify which parallel dimension causes the most overhead

### Phase 3: Compute Efficiency Analysis

Calculate theoretical FLOPs and MFU:

```python
# Forward pass FLOPs per iteration
forward_flops = 6 * batch_size * seq_len * params_count

# Total FLOPs (forward + backward)
total_flops = 2 * forward_flops  # backward is ~2x forward

# MFU calculation
measured_throughput = tokens_per_second
peak_flops = gpu_peak_flops * gpu_count
mfu = (measured_throughput * 6 * params_count) / peak_flops
```

### Phase 4: Profiling Data Integration

If profiling data is provided:

1. **Parse profiling trace:**
   - Identify memory spikes
   - Find communication hotspots
   - Detect load imbalance

2. **Bottleneck identification:**
   - Memory-bound: If activation memory > 50% total
   - Communication-bound: If comm_time > compute_time
   - Compute-bound: If MFU > 50% and low comm overhead

3. **Correlate with parallel strategy:**
   - High TP communication → Reduce TP size, increase DP
   - High activation memory → Enable SP or increase TP
   - Pipeline bubbles → Adjust PP size or micro-batch

### Phase 5: Strategy Recommendation

Generate candidate configurations and rank them:

**Candidate generation rules:**
- Total parallelism degree: dp_size * tp_size * pp_size * sp_size = total_gpus
- Constraints: tp_size <= 8 (typically), pp_size >= 2 for large models
- SP only beneficial when seq_len > 8192

**Scoring criteria (weighted):**
1. Throughput (tokens/sec) - 40%
2. MFU - 30%
3. Memory headroom - 20%
4. Communication efficiency - 10%

## Output Format

Generate a comprehensive Markdown report:

```markdown
# Distributed Training Parallel Strategy Analysis Report

## Executive Summary
- Recommended configuration: DP=x, TP=y, PP=z, SP=w
- Expected throughput: X tokens/sec
- Expected MFU: Y%
- Memory utilization: Z%

## Model & Hardware Context
[Summary of inputs]

## Memory Analysis
| Strategy | Model Memory | Activation Memory | Total/ GPU | Status |
|----------|-------------|-------------------|------------|--------|
| DP=8,TP=1,PP=1 | X GB | Y GB | Z GB | ✓/✗ |

## Communication Analysis
[Communication overhead breakdown]

## Profiling Insights
[If profiling data provided]

## Recommended Configurations

### Option 1: Best Throughput (Recommended)
- Configuration: DP=X, TP=Y, PP=Z, SP=W
- Rationale: ...
- Expected performance: ...

### Option 2: Memory Safe
[For models close to OOM limit]

### Option 3: Communication Optimized
[For low-bandwidth interconnects]

## Implementation Guide
```python
# Megatron-LM
--tensor-model-parallel-size Y
--pipeline-model-parallel-size Z
--sequence-parallel \
--distributed-backend nccl

# DeepSpeed
"tensor_parallel": {"tp_size": Y},
"pipeline_parallel": {"pp_size": Z}

# vLLM / SGLang
--tensor-parallel-size Y
--pipeline-parallel-size Z
```

## Risk Assessment
- OOM risk: Low/Medium/High
- Communication bottleneck: Yes/No
- Load imbalance risk: Low/Medium/High
```

## SFT-Specific Considerations

For Supervised Fine-Tuning scenarios:

1. **Sequence Length Handling:**
   - SFT typically has variable-length sequences
   - Packing strategy affects effective batch size
   - Consider SP when avg_seq_len > 8192

2. **Memory Pattern:**
   - SFT often has larger activation memory due to longer completions
   - Right-padding vs left-padding affects memory

3. **Batch Size Constraints:**
   - Micro-batch size often limited by longest sequence
   - Gradient accumulation more critical in SFT

## Special Cases

### MoE Models
- Add Expert Parallel (EP) dimension
- EP size typically equals number of experts or divides it
- Consider EP=1 for small expert counts (<8)

### Multi-Modal Models
- Vision encoder often uses different parallelism
- Consider decoupled TP for vision vs language

### Long Context (>32K)
- SP becomes essential
- Consider Ring Attention for >100K sequences
- PP may cause significant bubble overhead

## Reference

For detailed formulas and algorithms, see [reference.md](reference.md).

For SFT-specific examples, see [examples.md](examples.md).
