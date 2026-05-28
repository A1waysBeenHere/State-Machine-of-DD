# SFT Parallel Strategy Analysis Examples

## Example 1: Llama-2-7B SFT on 8x A100-80GB

### Input Configuration

```yaml
model:
  name: "Llama-2-7B"
  params: 6.7e9
  hidden_size: 4096
  num_layers: 32
  num_attention_heads: 32
  vocab_size: 32000
  intermediate_size: 11008

training:
  task: "SFT"
  avg_prompt_length: 512
  avg_completion_length: 2048
  max_seq_length: 4096
  micro_batch_size: 1
  gradient_accumulation_steps: 8
  precision: "bf16"
  activation_checkpointing: "selective"

hardware:
  gpu_type: "A100-80GB"
  gpu_memory: 80e9  # 80 GB
  num_gpus: 8
  gpus_per_node: 8
  nvlink_bandwidth: 600  # GB/s
  interconnect: "nvlink"

profiling:
  trace_file: "llama2_7b_sft_profile.json"
```

### Analysis Process

**Step 1: Memory Analysis**

```python
# Model parameters memory
param_memory = 6.7e9 * 2  # bf16 = 13.4 GB

# Optimizer states (Adam, fp32)
optimizer_memory = 6.7e9 * 2 * 4  # = 53.6 GB

# Gradients
grad_memory = 6.7e9 * 2  # = 13.4 GB

# Total sharded memory (before DP)
model_states = param_memory + optimizer_memory + grad_memory  # = 80.4 GB

# For different DP sizes:
# DP=1: 80.4 GB / 1 = 80.4 GB (too close to limit)
# DP=2: 80.4 GB / 2 = 40.2 GB per GPU
# DP=4: 80.4 GB / 4 = 20.1 GB per GPU
# DP=8: 80.4 GB / 8 = 10.05 GB per GPU

# Activation memory (seq_len=4096, batch=1, TP=1)
activation_memory = calculate_activation_memory(
    batch_size=1,
    seq_len=4096,
    hidden_size=4096,
    num_layers=32,
    tp_size=1,
    checkpointing="selective"
)  # ~12 GB

# Total per GPU for DP=8, TP=1:
# 10.05 + 12 = 22.05 GB (safe, 28% utilization)
```

**Step 2: Configuration Search**

Valid configurations for 8 GPUs:

| Config | DP | TP | PP | SP | Model/GPU | Act/GPU | Total/GPU | Status |
|--------|----|----|----|----|-----------|---------|-----------|--------|
| 1 | 8 | 1 | 1 | 1 | 10.1 GB | 12.0 GB | 22.1 GB | ✓ |
| 2 | 4 | 2 | 1 | 1 | 20.1 GB | 6.0 GB | 26.1 GB | ✓ |
| 3 | 4 | 2 | 1 | 2 | 20.1 GB | 3.0 GB | 23.1 GB | ✓ |
| 4 | 2 | 4 | 1 | 1 | 40.2 GB | 3.0 GB | 43.2 GB | ✓ |
| 5 | 2 | 2 | 2 | 1 | 40.2 GB | 6.0 GB | 46.2 GB | ✓ |
| 6 | 1 | 8 | 1 | 1 | 80.4 GB | 1.5 GB | 81.9 GB | ✗ OOM |

**Step 3: Communication Analysis**

```python
# Config 1: DP=8, TP=1
# DP all-reduce: 2*(8-1)/8 * 13.4 GB = 23.5 GB communication
# Time = 23.5 GB / 600 GB/s = 0.039 s

# Config 2: DP=4, TP=2
# DP all-reduce: 2*(4-1)/4 * 26.8 GB = 40.2 GB
# TP all-reduce: 2*(2-1)/2 * (1*4096*4096*2*2*32) = ~4.3 GB
# Total comm = 44.5 GB, Time = 0.074 s

# Config 4: DP=2, TP=4
# Higher TP = more communication overhead
# TP comm ~ 8.6 GB per layer pair
```

**Step 4: Profiling Integration**

```python
# From profiling trace:
profiling_insights = {
    "memory_peak": 45.2,  # GB
    "memory_pattern": "steady with spikes at layer transitions",
    "comm_time_pct": 15,  # % of total time
    "compute_time_pct": 75,
    "bubble_time_pct": 10,  # PP only
    "bottleneck": "communication_bound",
    "recommendation": "reduce_tp_increase_dp"
}
```

### Recommended Configuration

**Best Overall: Config 1 (DP=8, TP=1, PP=1, SP=1)**

```markdown
Rationale:
- For 7B model on 8x A100, model fits easily in single GPU memory
- No need for TP/PP which add communication overhead
- DP=8 provides best throughput with minimal communication
- Memory utilization at safe 28%

Expected Performance:
- Throughput: ~1800 tokens/sec
- MFU: ~45%
- Memory headroom: 72%
```

**Alternative for Longer Sequences: Config 3 (DP=4, TP=2, SP=2)**

```markdown
When seq_len > 8192, use SP to reduce activation memory:
- SP splits sequence across GPUs
- Reduces activation memory by 2x
- Allows larger micro-batch size
```

---

## Example 2: Llama-2-70B SFT on 16x A100-80GB

### Input Configuration

```yaml
model:
  name: "Llama-2-70B"
  params: 68.9e9
  hidden_size: 8192
  num_layers: 80
  num_attention_heads: 64
  intermediate_size: 28672

training:
  task: "SFT"
  avg_seq_length: 8192
  micro_batch_size: 1
  gradient_accumulation_steps: 4
  precision: "bf16"
  activation_checkpointing: "selective"

hardware:
  gpu_type: "A100-80GB"
  num_gpus: 16
  gpus_per_node: 8
  num_nodes: 2
  nvlink_bandwidth: 600
  ib_bandwidth: 200  # InfiniBand between nodes
```

### Analysis Process

**Step 1: Memory Analysis**

```python
# Model states (bf16 + Adam fp32)
param_memory = 68.9e9 * 2  # = 137.8 GB
optimizer_memory = 68.9e9 * 2 * 4  # = 551.2 GB
grad_memory = 68.9e9 * 2  # = 137.8 GB
model_states_total = 826.8 GB

# Per GPU with different sharding:
# DP=1, TP=16: 826.8/16 = 51.7 GB per GPU
# DP=2, TP=8: 826.8/16 = 51.7 GB per GPU  
# DP=4, TP=4: 826.8/16 = 51.7 GB per GPU

# Activation memory (seq_len=8192, TP reduces this)
# Without TP: ~48 GB activations
# With TP=4: ~12 GB activations
# With TP=8: ~6 GB activations

# Total memory requirements:
# Config DP=2, TP=8: 51.7 + 6 = 57.7 GB (safe)
# Config DP=4, TP=4: 51.7 + 12 = 63.7 GB (safe)
```

**Step 2: Communication Analysis**

Key consideration: Cross-node communication via InfiniBand (200 GB/s) is slower than NVLink (600 GB/s).

```python
# Strategy: Minimize cross-node TP communication
# TP within node (8 GPUs), DP across nodes

# Config: DP=2, TP=8, PP=1
# TP all-reduce happens within node (NVLink)
# DP all-reduce across nodes (IB)

# TP communication (within node):
tp_comm = calculate_tp_communication_time(
    batch_size=1, seq_len=8192, hidden_size=8192,
    num_layers=80, tp_size=8, bandwidth_gbps=600
)  # ~0.15s

# DP communication (across nodes):
dp_comm = calculate_dp_communication_time(
    params_count=68.9e9/8, dp_size=2, bandwidth_gbps=200
)  # ~0.55s
```

**Step 3: Pipeline Parallel Consideration**

```python
# Alternative: Use PP to reduce activation memory
# Config: DP=2, TP=4, PP=2

# Benefits:
# - Each stage has half the layers
# - Activation memory reduced
# - Can potentially increase batch size

# Costs:
# - Pipeline bubble overhead
# - P2P communication between stages

bubble_overhead = calculate_pp_bubble_overhead(
    pp_size=2, num_microbatches=4,
    forward_time=0.3, backward_time=0.6
)  # ~15% overhead
```

### Recommended Configuration

**Best Overall: DP=2, TP=8, PP=1, SP=1**

```markdown
Configuration Details:
- TP=8 uses full NVLink bandwidth within each node
- DP=2 minimizes cross-node communication
- No PP avoids bubble overhead
- Fits comfortably in 80GB memory

Expected Performance:
- Throughput: ~850 tokens/sec
- MFU: ~38%
- Memory utilization: 72%
```

**Alternative with PP: DP=2, TP=4, PP=2**

```markdown
Use when:
- Sequence length > 16384 (activation memory concern)
- Need to support larger batch sizes
- Accept 10-15% bubble overhead
```

---

## Example 3: Qwen-72B Long Context SFT on 32x H100-80GB

### Input Configuration

```yaml
model:
  name: "Qwen-72B"
  params: 72.7e9
  hidden_size: 8192
  num_layers: 80
  num_attention_heads: 64
  intermediate_size: 22016
  use_sliding_window: true
  max_position_embeddings: 32768

training:
  task: "SFT"
  avg_seq_length: 16384  # Long context
  micro_batch_size: 1
  gradient_accumulation_steps: 2
  precision: "bf16"
  activation_checkpointing: "full"

hardware:
  gpu_type: "H100-80GB"
  num_gpus: 32
  gpus_per_node: 8
  num_nodes: 4
  nvlink_bandwidth: 900
  ib_bandwidth: 400
```

### Analysis Process

**Step 1: Memory Analysis with Long Context**

```python
# Long context = massive activation memory
# Without SP: activation_memory ~ 96 GB (OOM!)
# With SP=8: activation_memory ~ 12 GB

# Model states (with TP=8):
model_states_per_gpu = (72.7e9 * 2 * 3) / 8  # = 54.5 GB

# Activations with SP=8:
activation_memory = calculate_activation_memory(
    batch_size=1, seq_len=16384, hidden_size=8192,
    num_layers=80, tp_size=8, sp_size=8,
    checkpointing="full"
)  # ~8 GB

# Total: 54.5 + 8 = 62.5 GB (safe on H100-80GB)
```

**Step 2: SP is Essential**

```python
# Without SP:
activation_no_sp = calculate_activation_memory(
    seq_len=16384, tp_size=8, sp_size=1
)  # ~64 GB
# Total: 54.5 + 64 = 118.5 GB -> OOM!

# With SP=8:
# Splits sequence dimension across TP group
# Reduces activation memory by 8x
```

**Step 3: Communication Analysis**

```python
# Config: DP=4, TP=8, PP=1, SP=8
# Total GPUs = 4 * 8 = 32

# SP communication cost:
sp_comm = calculate_sp_communication_time(
    batch_size=1, seq_len=16384, hidden_size=8192,
    num_layers=80, sp_size=8, bandwidth_gbps=900
)  # ~0.12s

# TP communication (overlaps with SP):
tp_comm = calculate_tp_communication_time(
    batch_size=1, seq_len=16384/8,  # SP reduces per-GPU seq len
    hidden_size=8192, num_layers=80,
    tp_size=8, bandwidth_gbps=900
)  # ~0.08s
```

### Recommended Configuration

**Long Context Optimized: DP=4, TP=8, PP=1, SP=8**

```markdown
Key Features:
- SP=8 is essential for 16K+ sequences
- TP=8 provides efficient all-reduce within node
- DP=4 scales across 4 nodes
- Full activation checkpointing for safety

Expected Performance:
- Throughput: ~620 tokens/sec
- MFU: ~35%
- Memory utilization: 78%
- Max supported seq_len: 32768
```

---

## Example 4: MoE Model (Mixtral 8x7B) SFT

### Input Configuration

```yaml
model:
  name: "Mixtral-8x7B"
  params: 46.7e9  # Active params: 12.9B
  num_experts: 8
  experts_per_token: 2
  hidden_size: 4096
  num_layers: 32
  intermediate_size: 14336

training:
  task: "SFT"
  avg_seq_length: 4096
  micro_batch_size: 2
  precision: "bf16"

hardware:
  gpu_type: "A100-80GB"
  num_gpus: 8
```

### Expert Parallel (EP) Analysis

```python
# MoE adds Expert Parallel dimension
# EP size should divide num_experts

# For 8 experts:
# EP=1: All experts on all GPUs (highest memory)
# EP=2: 4 experts per GPU group
# EP=4: 2 experts per GPU group
# EP=8: 1 expert per GPU (highest communication)

# All-to-all communication for EP:
def calculate_ep_communication(
    batch_size, seq_len, hidden_size, ep_size, bandwidth_gbps
):
    # Each token routes to 2 experts
    tokens_per_expert = batch_size * seq_len * 2 / ep_size
    comm_volume = tokens_per_expert * hidden_size * 2  # all-to-all
    return comm_volume / (bandwidth_gbps * 1e9 / 8)

# EP=8 on 8 GPUs:
# Each GPU holds 1 expert
# All-to-all communication across all GPUs
```

### Recommended Configuration

**MoE Optimized: DP=1, TP=1, PP=1, EP=8**

```markdown
For MoE models:
- EP=8 distributes experts across all GPUs
- Minimizes per-GPU memory for expert params
- All-to-all overhead acceptable for 8 GPUs
- Consider EP=4 + DP=2 for larger clusters

Expected Performance:
- Throughput: ~1400 tokens/sec
- MFU: ~32% (lower due to all-to-all)
- Memory utilization: 65%
```

---

## Profiling Data Integration Examples

### Example: Memory Spike Detection

```json
{
  "trace_events": [
    {"name": "allocate", "ph": "B", "ts": 1000, "args": {"size": 1073741824}},
    {"name": "allocate", "ph": "E", "ts": 1005, "args": {"size": 1073741824}},
    {"name": "forward", "ph": "B", "ts": 1010},
    {"name": "forward", "ph": "E", "ts": 1500},
    {"name": "backward", "ph": "B", "ts": 1505},
    {"name": "backward", "ph": "E", "ts": 2500}
  ],
  "memory_stats": {
    "peak_allocated": 68719476736,
    "peak_reserved": 75161927680
  }
}
```

Analysis:
- Peak memory: 64 GB allocated, 70 GB reserved
- Forward pass: 490ms
- Backward pass: 995ms (~2x forward, expected)
- No significant memory spikes detected

### Example: Communication Bottleneck

```json
{
  "communication_stats": {
    "all_reduce_count": 256,
    "all_reduce_total_time_ms": 450,
    "all_gather_count": 64,
    "all_gather_total_time_ms": 120,
    "p2p_count": 0
  },
  "compute_stats": {
    "forward_time_ms": 850,
    "backward_time_ms": 1700
  }
}
```

Analysis:
- Communication time: 570ms
- Compute time: 2550ms
- Comm/Compute ratio: 0.22 (22%)
- **Finding**: Communication bound detected
- **Recommendation**: Reduce TP size, increase DP

### Example: Load Imbalance

```json
{
  "pipeline_stats": {
    "stage_0_forward_ms": 210,
    "stage_1_forward_ms": 215,
    "stage_2_forward_ms": 208,
    "stage_3_forward_ms": 450,  // Outlier!
    "bubble_time_ms": 180
  }
}
```

Analysis:
- Stage 3 takes 2x longer than other stages
- Indicates load imbalance in PP
- Possible causes:
  - Uneven layer distribution
  - Different input sizes per stage
- **Recommendation**: Rebalance layers or check input distribution
