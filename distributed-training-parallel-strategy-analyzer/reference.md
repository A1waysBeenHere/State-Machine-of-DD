# Parallel Strategy Analysis Reference

## Memory Calculation Formulas

### 1. Model Parameters Memory

```python
def calculate_param_memory(params_count, precision="bf16"):
    """
    Calculate memory for model parameters.
    
    Args:
        params_count: Total model parameters (e.g., 7e9 for 7B)
        precision: "fp32", "fp16", "bf16"
    
    Returns:
        Memory in bytes
    """
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2
    }
    return params_count * bytes_per_param[precision]
```

### 2. Optimizer States Memory

```python
def calculate_optimizer_memory(params_count, optimizer="adam", precision="bf16"):
    """
    Calculate optimizer state memory.
    
    Adam: stores momentum (fp32) + variance (fp32) per parameter
    SGD: stores momentum only
    """
    param_mem = calculate_param_memory(params_count, precision)
    
    if optimizer == "adam":
        # Momentum + variance, both fp32
        return param_mem * 2 * (4 / bytes_per_param[precision])
    elif optimizer == "sgd":
        return param_mem * (4 / bytes_per_param[precision])
    return 0
```

### 3. Activation Memory

```python
def calculate_activation_memory(
    batch_size,
    seq_len,
    hidden_size,
    num_layers,
    tp_size=1,
    sp_size=1,
    checkpointing="none",
    precision="bf16"
):
    """
    Calculate activation memory per GPU.
    
    Args:
        checkpointing: "none", "selective", "full"
    """
    bytes_per_activation = 2 if precision in ["fp16", "bf16"] else 4
    
    # Base activation per layer (simplified)
    # Includes: input, attention Q/K/V, attention output, MLP intermediate, MLP output
    activations_per_layer = (
        batch_size * seq_len * hidden_size * 2 +  # Input + attention input
        batch_size * seq_len * hidden_size * 3 +  # Q, K, V
        batch_size * seq_len * hidden_size +      # Attention output
        batch_size * seq_len * hidden_size * 4 +  # MLP intermediate (assuming 4x)
        batch_size * seq_len * hidden_size        # MLP output
    )
    
    # Apply TP/SP reduction
    activations_per_layer /= (tp_size * sp_size)
    
    # Apply checkpointing
    if checkpointing == "full":
        # Only store input to each transformer layer
        total_activations = batch_size * seq_len * hidden_size * num_layers
    elif checkpointing == "selective":
        # Store selective activations, ~1/3 of full
        total_activations = activations_per_layer * num_layers * 0.33
    else:
        total_activations = activations_per_layer * num_layers
    
    return total_activations * bytes_per_activation
```

### 4. Communication Buffer Memory

```python
def calculate_communication_memory(
    hidden_size,
    seq_len,
    batch_size,
    tp_size=1,
    pp_size=1,
    precision="bf16"
):
    """
    Estimate communication buffer memory.
    """
    bytes_per_param = 2 if precision in ["fp16", "bf16"] else 4
    
    # TP all-reduce buffers
    tp_buffer = 0
    if tp_size > 1:
        # Need buffers for all-reduce in attention and MLP
        tp_buffer = 2 * batch_size * seq_len * hidden_size * bytes_per_param
    
    # PP send/recv buffers
    pp_buffer = 0
    if pp_size > 1:
        # Buffers for activation send/recv between pipeline stages
        pp_buffer = 2 * batch_size * seq_len * hidden_size * bytes_per_param
    
    return tp_buffer + pp_buffer
```

## Communication Cost Formulas

### 1. Data Parallel (DP) Communication

```python
def calculate_dp_communication_time(
    params_count,
    dp_size,
    bandwidth_gbps,
    precision="bf16"
):
    """
    Calculate DP all-reduce time for gradients.
    
    Uses ring all-reduce: 2*(N-1)/N * data_size
    """
    bytes_per_param = 2 if precision in ["fp16", "bf16"] else 4
    gradient_size = params_count * bytes_per_param
    
    # Ring all-reduce communication volume
    comm_volume = 2 * (dp_size - 1) / dp_size * gradient_size
    
    # Time = volume / bandwidth
    bandwidth_bytes_per_sec = bandwidth_gbps * 1e9 / 8
    return comm_volume / bandwidth_bytes_per_sec
```

### 2. Tensor Parallel (TP) Communication

```python
def calculate_tp_communication_time(
    batch_size,
    seq_len,
    hidden_size,
    num_layers,
    tp_size,
    bandwidth_gbps,
    precision="bf16"
):
    """
    Calculate TP all-reduce time.
    
    TP performs all-reduce in:
    - Attention (after softmax)
    - MLP (after second linear)
    """
    bytes_per_param = 2 if precision in ["fp16", "bf16"] else 4
    
    # Activation size per all-reduce
    activation_size = batch_size * seq_len * hidden_size * bytes_per_param
    
    # 2 all-reduces per layer (attention + MLP)
    # Each all-reduce: (N-1)/N * data_size for ring algorithm
    comm_volume_per_layer = 2 * (tp_size - 1) / tp_size * activation_size
    
    total_comm_volume = comm_volume_per_layer * num_layers
    
    bandwidth_bytes_per_sec = bandwidth_gbps * 1e9 / 8
    return total_comm_volume / bandwidth_bytes_per_sec
```

### 3. Pipeline Parallel (PP) Communication

```python
def calculate_pp_communication_time(
    batch_size,
    seq_len,
    hidden_size,
    pp_size,
    bandwidth_gbps,
    precision="bf16"
):
    """
    Calculate PP P2P communication time.
    
    PP sends/receives activations between stages.
    """
    bytes_per_param = 2 if precision in ["fp16", "bf16"] else 4
    
    # Activation size per stage boundary
    activation_size = batch_size * seq_len * hidden_size * bytes_per_param
    
    # Forward: send to next stage
    # Backward: send grad to prev stage
    # Total 2 transfers per batch
    comm_volume = 2 * activation_size
    
    bandwidth_bytes_per_sec = bandwidth_gbps * 1e9 / 8
    return comm_volume / bandwidth_bytes_per_sec
```

### 4. Sequence Parallel (SP) Communication

```python
def calculate_sp_communication_time(
    batch_size,
    seq_len,
    hidden_size,
    num_layers,
    sp_size,
    bandwidth_gbps,
    precision="bf16"
):
    """
    Calculate SP all-gather/reduce-scatter time.
    
    SP splits sequence dimension, needs communication for attention.
    """
    bytes_per_param = 2 if precision in ["fp16", "bf16"] else 4
    
    # All-gather Q, K, V
    qkv_size = 3 * batch_size * seq_len * hidden_size * bytes_per_param
    
    # Reduce-scatter attention output
    output_size = batch_size * seq_len * hidden_size * bytes_per_param
    
    # Communication volume per layer
    # All-gather: (N-1)/N * data_size
    # Reduce-scatter: (N-1)/N * data_size
    comm_volume_per_layer = (sp_size - 1) / sp_size * (qkv_size + output_size)
    
    total_comm_volume = comm_volume_per_layer * num_layers
    
    bandwidth_bytes_per_sec = bandwidth_gbps * 1e9 / 8
    return total_comm_volume / bandwidth_bytes_per_sec
```

## Compute Time Estimation

```python
def calculate_compute_time(
    batch_size,
    seq_len,
    params_count,
    gpu_flops,
    mfu_estimate=0.5
):
    """
    Estimate compute time per iteration.
    
    Args:
        gpu_flops: Peak FLOPs of GPU (e.g., 312e12 for A100)
        mfu_estimate: Expected MFU (0.0 - 1.0)
    """
    # Forward pass: ~2 * params * tokens
    forward_flops = 2 * params_count * batch_size * seq_len
    
    # Backward pass: ~2x forward
    total_flops = 3 * forward_flops
    
    effective_flops = gpu_flops * mfu_estimate
    return total_flops / effective_flops
```

## Pipeline Bubble Calculation

```python
def calculate_pp_bubble_overhead(
    pp_size,
    num_microbatches,
    forward_time,
    backward_time
):
    """
    Calculate pipeline bubble overhead.
    
    Bubble = (pp_size - 1) / num_microbatches * (forward_time + backward_time)
    """
    if pp_size == 1:
        return 0
    
    bubble_time = (pp_size - 1) / num_microbatches * (forward_time + backward_time)
    total_time = forward_time + backward_time + bubble_time
    
    return bubble_time / total_time  # Return as fraction
```

## Configuration Search Space

```python
def generate_valid_configurations(
    total_gpus,
    min_tp=1,
    max_tp=8,
    min_pp=1,
    max_pp=16,
    enable_sp=True
):
    """
    Generate all valid parallel configurations.
    
    Constraints:
    - dp * tp * pp * sp = total_gpus
    - tp in powers of 2 (typically)
    - pp >= 1
    - sp divides tp (usually sp == tp or sp == 1)
    """
    configs = []
    
    for tp in [1, 2, 4, 8]:
        if tp < min_tp or tp > max_tp:
            continue
        
        for pp in range(min_pp, max_pp + 1):
            remaining = total_gpus // (tp * pp)
            
            if tp * pp * remaining != total_gpus:
                continue
            
            # SP options
            sp_options = [1]
            if enable_sp and tp > 1:
                sp_options.append(tp)  # SP usually equals TP
            
            for sp in sp_options:
                if tp * pp * sp > total_gpus:
                    continue
                dp = total_gpus // (tp * pp * sp)
                
                configs.append({
                    "dp": dp,
                    "tp": tp,
                    "pp": pp,
                    "sp": sp
                })
    
    return configs
```

## Scoring Function

```python
def score_configuration(
    config,
    throughput,
    mfu,
    memory_utilization,
    comm_overhead,
    weights=None
):
    """
    Score a configuration based on multiple factors.
    
    Higher score = better configuration.
    """
    if weights is None:
        weights = {
            "throughput": 0.4,
            "mfu": 0.3,
            "memory_headroom": 0.2,
            "comm_efficiency": 0.1
        }
    
    # Normalize metrics (0-1 scale)
    throughput_score = min(throughput / 10000, 1.0)  # Normalize to 10K tokens/sec
    mfu_score = mfu
    memory_headroom_score = 1.0 - memory_utilization  # Higher headroom = better
    comm_efficiency_score = 1.0 - comm_overhead
    
    # Weighted sum
    total_score = (
        weights["throughput"] * throughput_score +
        weights["mfu"] * mfu_score +
        weights["memory_headroom"] * memory_headroom_score +
        weights["comm_efficiency"] * comm_efficiency_score
    )
    
    return total_score
```

## GPU Specifications Reference

| GPU | Memory | Peak FP16 TFLOPS | Peak BF16 TFLOPS | NVLink Bandwidth |
|-----|--------|------------------|------------------|------------------|
| A100-40GB | 40 GB | 312 | 312 | 600 GB/s |
| A100-80GB | 80 GB | 312 | 312 | 600 GB/s |
| H100-80GB | 80 GB | 989 | 989 | 900 GB/s |
| H100-NVL | 188 GB | 989 | 989 | 900 GB/s |
| A10-24GB | 24 GB | 125 | 125 | No NVLink |
| L40S-48GB | 48 GB | 183 | 183 | No NVLink |
| MI250X | 128 GB | 383 | 383 | 800 GB/s |
| MI300X | 192 GB | 1300 | 1300 | 896 GB/s |

## Model Architecture Reference

### Llama Family

| Model | Params | Hidden | Layers | Heads | Head Dim | MLP Ratio |
|-------|--------|--------|--------|-------|----------|-----------|
| Llama-7B | 6.7B | 4096 | 32 | 32 | 128 | 2.7x |
| Llama-13B | 13.0B | 5120 | 40 | 40 | 128 | 2.7x |
| Llama-30B | 32.5B | 6656 | 60 | 52 | 128 | 2.7x |
| Llama-65B | 65.2B | 8192 | 80 | 64 | 128 | 2.7x |
| Llama-2-7B | 6.7B | 4096 | 32 | 32 | 128 | 2.7x |
| Llama-2-13B | 13.0B | 5120 | 40 | 40 | 128 | 2.7x |
| Llama-2-70B | 68.9B | 8192 | 80 | 64 | 128 | 3.5x |
| Llama-3-8B | 8.0B | 4096 | 32 | 32 | 128 | 3x |
| Llama-3-70B | 70.0B | 8192 | 80 | 64 | 128 | 3.5x |

### GPT Family

| Model | Params | Hidden | Layers | Heads | MLP Ratio |
|-------|--------|--------|--------|-------|-----------|
| GPT-3-1.3B | 1.3B | 2048 | 24 | 16 | 4x |
| GPT-3-2.7B | 2.7B | 2560 | 32 | 32 | 4x |
| GPT-3-6.7B | 6.7B | 4096 | 32 | 32 | 4x |
| GPT-3-13B | 13.0B | 5140 | 40 | 40 | 4x |
| GPT-3-175B | 175.0B | 12288 | 96 | 96 | 4x |

### Qwen Family

| Model | Params | Hidden | Layers | Heads | MLP Ratio |
|-------|--------|--------|--------|-------|-----------|
| Qwen-7B | 7.7B | 4096 | 32 | 32 | 2.7x |
| Qwen-14B | 14.2B | 5120 | 40 | 40 | 2.7x |
| Qwen-72B | 72.7B | 8192 | 80 | 64 | 2.7x |
