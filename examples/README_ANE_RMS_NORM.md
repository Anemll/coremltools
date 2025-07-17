# ANE RMS Norm Implementation

This directory contains examples demonstrating the implementation and usage of ANE (Apple Neural Engine) RMS Norm optimization in Core ML Tools.

## Overview

RMS Norm (Root Mean Square Normalization) is a simplified version of Layer Normalization that normalizes by the RMS of the input rather than the mean and variance. This implementation provides:

1. **Automatic PyTorch Conversion**: PyTorch `RMSNorm` modules are automatically converted to `ane_rms_norm` operations
2. **ANE Optimization**: The `ane_rms_norm` operations are lowered to primitive ops optimized for Apple Neural Engine
3. **Compute Unit Filtering**: Optimizations are only applied when compute units include the Neural Engine

## Files

- `pytorch_rms_norm_to_ane_example.py` - Complete pipeline from PyTorch to ANE-optimized Core ML
- `ane_rms_norm_compute_units_example.py` - Demonstrates compute unit filtering behavior

## Quick Start

### Basic PyTorch RMSNorm Usage

```python
import torch
import torch.nn as nn
import coremltools as ct

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        rms = x.norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
        x_normed = x / (rms + self.eps)
        return self.weight * x_normed

# Create model
model = RMSNorm(512)
model.eval()

# Create example input and trace model
example_input = torch.randn(1, 128, 512)
traced_model = torch.jit.trace(model, example_input)

# Convert to Core ML with ANE optimization
mlmodel = ct.convert(
    traced_model,
    source="pytorch",
    inputs=[ct.TensorType(shape=(1, 128, 512))],
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.iOS15,
)
```

### Automatic Conversion Pipeline

1. **PyTorch RMSNorm** → **ane_rms_norm** (during conversion)
2. **ane_rms_norm** → **Primitive ops** (negative, concat, layer_norm, slice_by_index, mul)

The conversion happens automatically when:
- Using `compute_units` that include Neural Engine (`CPU_AND_NE` or `ALL`)
- The PyTorch model contains RMSNorm patterns

### Compute Unit Behavior

| Compute Unit | ANE RMS Norm Pass | Behavior |
|--------------|------------------|----------|
| `CPU_ONLY` | Skipped | Uses standard layer_norm |
| `CPU_AND_GPU` | Skipped | Uses standard layer_norm |
| `CPU_AND_NE` | Applied | Lowers to ANE-optimized ops |
| `ALL` | Applied | Lowers to ANE-optimized ops |

## Advanced Usage

### Custom Pass Pipeline

```python
from coremltools.converters.mil.mil.passes.pass_pipeline import PassPipeline

# Create custom pipeline
pipeline = PassPipeline()
pipeline.append_pass("common::lower_ane_rms_norm_to_layer_norm")

# Set compute unit option
pipeline.set_options("common::lower_ane_rms_norm_to_layer_norm", {
    "compute_units": ct.ComputeUnit.CPU_AND_NE
})

# Create example input and trace model
example_input = torch.randn(1, 128, 512)
traced_model = torch.jit.trace(model, example_input)

# Convert with custom pipeline
mlmodel = ct.convert(
    traced_model,
    source="pytorch",
    inputs=[ct.TensorType(shape=(1, 128, 512))],
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    pass_pipeline=pipeline,
)
```

### Explicit Pass Application

```python
from coremltools.converters.mil.mil.passes.defs.ane_rms_norm_to_layer_norm import lower_ane_rms_norm_to_layer_norm

# Get MIL program
mil_program = mlmodel._mil_program

# Apply pass explicitly
pass_instance = lower_ane_rms_norm_to_layer_norm()
pass_instance.compute_units = ct.ComputeUnit.CPU_AND_NE
pass_instance.apply(mil_program)

# Convert back to Core ML
lowered_mlmodel = ct.convert(
    mil_program,
    convert_from="milinternal",
    convert_to="mlprogram",
    compute_units=ct.ComputeUnit.CPU_AND_NE,
)
```

## Performance Benefits

### ANE Optimization Benefits

1. **Faster Inference**: Optimized for Apple Neural Engine architecture
2. **Lower Power Consumption**: More efficient execution on Apple Silicon
3. **Better Performance**: Leverages specialized ANE hardware
4. **Automatic Optimization**: No manual intervention required

### When to Use

- **Use ANE optimization** when targeting Apple Silicon devices (iPhone, iPad, Mac with Apple Silicon)
- **Use CPU only** for debugging or when ANE is not available
- **Use ALL compute units** for maximum performance on Apple devices

## Technical Details

### RMS Norm Implementation

The ANE RMS Norm is implemented using a clever trick with LayerNorm:

1. **Input**: `x` tensor
2. **Negative**: Create `-x`
3. **Concat**: Concatenate `[x, -x]` along the feature dimension
4. **LayerNorm**: Apply layer normalization to the concatenated tensor
5. **Slice**: Take the first half (original `x` part)
6. **Multiply**: Apply learnable weight `gamma`

This approach allows RMS Norm to be executed efficiently on ANE using existing LayerNorm hardware.

### Mathematical Equivalence

The ANE implementation is mathematically equivalent to standard RMS Norm:

```
Standard RMS Norm:
rms = sqrt(mean(x²))
output = gamma * (x / (rms + eps))

ANE Implementation:
concat = [x, -x]
normalized = layer_norm(concat)
output = gamma * slice(normalized, first_half)
```

## Troubleshooting

### Common Issues

1. **ane_rms_norm not detected**: Ensure you're using `compute_units` that include Neural Engine
2. **Pass not applied**: Check that the compute unit is `CPU_AND_NE` or `ALL`
3. **Performance not improved**: Verify the model is running on Apple Silicon hardware

### Debugging

```python
# Check operations in converted model
mil_program = mlmodel._mil_program
operations = get_op_types_in_program(mil_program)
print(f"Operations: {operations}")

# Check if ane_rms_norm is present
if "ane_rms_norm" in operations:
    print("ANE RMS Norm detected!")
    
# Check if it was lowered
primitive_ops = ["negative", "concat", "layer_norm", "slice_by_index", "mul"]
if all(op in operations for op in primitive_ops):
    print("ANE RMS Norm was lowered to primitive ops!")
```

## Examples

Run the examples to see the complete pipeline in action:

```bash
# PyTorch to ANE pipeline
python pytorch_rms_norm_to_ane_example.py

# Compute unit filtering
python ane_rms_norm_compute_units_example.py
```

## Requirements

- Core ML Tools 8.0+
- PyTorch 2.0+
- iOS 15+ deployment target (for ane_rms_norm support)
- Apple Silicon device for optimal performance

## References

- [Core ML Tools Documentation](https://apple.github.io/coremltools/)
- [Apple Neural Engine](https://developer.apple.com/machine-learning/neural-engine/)
- [RMS Norm Paper](https://arxiv.org/abs/1910.07467) 