# FP8 Quantization

FP8 (8-bit floating point) quantization stores weights and activations in the E4M3 or E5M2
formats. On a Mac with an M6 chip or later, the Neural Engine runs FP8 (E4M3) weights and
activations, which can be faster than FP16 and uses half the weight bytes.

FP8 lives in the iOS 26 opset (Core ML 9, specification version 10). There is no newer target.
Use `ct.target.iOS26` / `ct.target.macOS26`.

```{admonition} Requirements

- macOS 27 (or later) to compile and run, and an **M6 or later** for the Neural Engine to run
  FP8. On other Macs FP8 falls back to a much slower CPU path.
- The `ml_dtypes` package, which provides the numpy FP8 dtypes coremltools uses:
  `pip install ml_dtypes`.
- A model converted with `minimum_deployment_target=ct.target.iOS26`, or an existing model that
  is upgraded to iOS 26 by the FP8 quantization API.
```

## How FP8 Quantization Works

FP8 is a symmetric (zero-point-free) linear quantization. Core ML's `quantize` op computes:

```
q = clip(round_to_nearest(x / scale), -FP8_MAX, FP8_MAX)
```

and `dequantize` computes `x = q * scale`. Out-of-range values saturate rather than become NaN,
and `zero_point` must not be set. FP8 uses the following ranges:

| dtype | Largest finite value | Neural Engine |
|---|---:|---|
| `fp8e4m3fn` | 448 | Yes (M6 and later) |
| `fp8e5m2` | 57344 | No (CPU only) |

Weights and activations are quantized separately:

- **Weights** are compressed with `linear_quantize_weights`, using
  `OpLinearQuantizerConfig(dtype="fp8e4m3fn")`.
- **Activations** are quantized with `linear_quantize_activations`, using calibration data to
  derive per-tensor scales.

### The Neural Engine reads FP8 weights like IEEE E4M3

The Neural Engine decodes FP8 *weights* as IEEE E4M3, whose largest finite value is **240**.
E4M3FN codes between 240 and 448 become infinity there. For that reason, FP8 weights are scaled
to ±240 by default (`fp8_max=None` defaults to 240 for `fp8e4m3fn`). Activations keep the full
±448 range, which the Neural Engine handles. Pass `fp8_max=448` only for CPU-only models.

## Quantizing Weights to FP8

```python
import coremltools as ct
import coremltools.optimize as cto

model = ct.models.MLModel("my_model.mlpackage")

config = cto.coreml.OptimizationConfig(
    global_config=cto.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric",       # FP8 is symmetric; this is the only allowed mode
        dtype="fp8e4m3fn",             # use "fp8e5m2" only for CPU-only models
        granularity="per_channel",
    )
)
model = cto.coreml.linear_quantize_weights(model, config)
```

`OpLinearQuantizerConfig` accepts these additional FP8 fields:

- **`fp8_max`** — caps the FP8 range so `scale = max(abs(x)) / fp8_max`. Defaults to `240` for
  `fp8e4m3fn` weights, `448` for `fp8e5m2`, and the full range for activations.
- **`fp8_encoding`** — how FP8 weights are stored:
  - `"blockwise"` (default): `constexpr_blockwise_shift_scale` with FP8 data, at any
    granularity. The Core ML compiler in macOS 27 crashes on these weights while collecting
    compile analytics. `ct.models.utils.compile_model` and `MLModel` compile around the crash,
    but Xcode cannot compile the `.mlpackage` — ship the compiled `.mlmodelc` instead.
  - `"palette"`: `constexpr_lut_to_dense` with the E4M3 values as a table and the scale applied
    around each `conv` / `linear` op. It compiles with the stock compiler, including Xcode, and
    the Neural Engine runs it as native FP8 weights. Supports `fp8e4m3fn` with `per_tensor` or
    `per_channel` granularity, for weights consumed only by `conv` or `linear` ops. The two extra
    `mul` ops per layer cost bandwidth when activations are large, so `"blockwise"` is faster
    there.

## Quantizing Activations to FP8

Use `linear_quantize_activations` with sample data to calibrate activation ranges. The
`global_config` must use the same FP8 dtype:

```python
activation_config = cto.coreml.OptimizationConfig(
    global_config=cto.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="fp8e4m3fn",
    )
)
model = cto.coreml.linear_quantize_activations(model, activation_config, sample_data)
```

`sample_data` is a list of input dictionaries in the same format as `.predict`. Apply weight
quantization after activation quantization to get an FP8 W8A8 model.

## Converting a Model to the iOS 26 Opset

New models should be converted directly to iOS 26:

```python
model = ct.convert(
    traced_model,
    minimum_deployment_target=ct.target.iOS26,
    compute_units=ct.ComputeUnit.CPU_AND_NE,
)
```

When you apply an FP8 config to a model with an older deployment target,
`linear_quantize_weights` and `linear_quantize_activations` upgrade it to iOS 26 automatically.
You can also write FP8 MIL directly with the builder:

```python
from coremltools.converters.mil import Builder as mb

@mb.program(input_specs=[...], opset_version=ct.target.iOS26)
def prog(x):
    q = mb.quantize(input=x, scale=scale, output_dtype="fp8e4m3fn")   # no zero_point
    x = mb.dequantize(input=q, scale=scale)
    ...
```

## Implementation Notes

- **Types.** `fp8e4m3fn` and `fp8e5m2` are MIL builtin types backed by the `ml_dtypes` float8
  numpy dtypes. The MIL proto uses `FLOAT8E4M3FN=40` and `FLOAT8E5M2=41`, and the weight blob
  format uses blob dtypes `16` / `17`.
- **Ops.** The iOS 26 versions of `quantize`, `dequantize`, and
  `constexpr_blockwise_shift_scale` accept FP8. FP8 is symmetric, so `zero_point` / `offset`
  are rejected.
- **Compile workaround.** The macOS 27 Core ML compiler (`coremlcompiler` and
  `MLModel.compileModel`) crashes on FP8 constexpr weights while collecting compile analytics
  (`-[__NSSetM addObject:]: object cannot be nil`). `coremltools.models._fp8_compile` compiles a
  copy of the package with int8 stand-ins and restores the FP8 dtype in the compiled `model.mil`
  and blob metadata. This is automatic in `ct.models.utils.compile_model` and `MLModel`.
- **Rounding of exact ties.** Converting to FP8 rounds to nearest. For values exactly halfway
  between two FP8 values, the CPU rounds toward `-inf`, the Neural Engine rounds away from zero,
  and coremltools (when it folds constants) rounds to even. Out-of-range values saturate to
  ±448 everywhere.
- **No GPU path.** The GPU has no FP8 path. With `.cpuAndGPU`, Core ML runs FP8 models on the
  CPU. Use `.cpuAndNE` or `.all`.

## Checking Placement

`MLComputePlan` reports where each op runs. In an FP8 model on an M6, `conv` / `linear`,
`quantize`, and `dequantize` should show `NeuralEngine`:

```python
from coremltools.models.compute_plan import MLComputePlan

plan = MLComputePlan.load_from_path(compiled_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
for op in plan.model_structure.program.functions["main"].block.operations:
    usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
    if usage is not None:
        print(op.operator_name, type(usage.preferred_compute_device).__name__)
```

## Learn More

The [`examples/fp8`](https://github.com/Anemll/coremltools/tree/fp8-ane-support/examples/fp8)
directory has runnable examples:

- `fp8_mil_example.py` — a 1x1 conv chain built with the MIL builder, FP8 weights and
  activations, placement and timing against FP16.
- `fp8_pytorch_example.py` — a PyTorch feed-forward block quantized with the API above.
- `fp8_resnet50.py` — ResNet50 on ImageNetV2, FP8 vs INT8 vs FP16 accuracy and latency.
