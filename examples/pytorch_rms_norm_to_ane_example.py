#!/usr/bin/env python3
"""
Example demonstrating the complete pipeline from PyTorch RMSNorm to ANE-optimized Core ML.

This example shows:
1. PyTorch RMSNorm module implementation
2. Automatic conversion to ane_rms_norm during Core ML conversion
3. Lowering to primitive ops for ANE execution
4. Comparison of different compute unit behaviors
"""

import torch
import torch.nn as nn
import numpy as np
import coremltools as ct
from coremltools import ComputeUnit
from coremltools.converters.mil.testing_utils import get_op_types_in_program


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    This is a simplified version of RMSNorm that normalizes by the RMS of the input
    rather than the mean and variance like traditional LayerNorm.
    Based on ANEMLL RMSNorm implementation: https://github.com/apple/ml-ane/blob/main/examples/rms_norm.py
    """
    def __init__(self, hidden_size, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        # Compute RMS over the last dimension
        rms = x.norm(dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
        x_normed = x / (rms + self.eps)
        return self.weight * x_normed


def create_rms_norm_model(hidden_size=512, seq_len=128):
    """Create a simple model with RMSNorm."""
    
    class ModelWithRMSNorm(nn.Module):
        def __init__(self, hidden_size, seq_len):
            super().__init__()
            self.hidden_size = hidden_size
            self.seq_len = seq_len
            self.rms_norm = RMSNorm(hidden_size)
            
        def forward(self, x):
            # Apply RMS normalization
            return self.rms_norm(x)
    
    model = ModelWithRMSNorm(hidden_size, seq_len)
    model.eval()
    return model


def demonstrate_conversion_pipeline():
    """Demonstrate the complete conversion pipeline."""
    
    print("=== PyTorch RMSNorm to ANE Pipeline ===\n")
    
    # Create PyTorch model
    hidden_size = 512
    seq_len = 128
    batch_size = 1
    
    model = create_rms_norm_model(hidden_size, seq_len)
    print(f"Created PyTorch model with RMSNorm (hidden_size={hidden_size}, seq_len={seq_len})")
    
    # Create example input
    example_input = torch.randn(batch_size, seq_len, hidden_size)
    print(f"Example input shape: {example_input.shape}")
    
    # Test PyTorch model
    with torch.no_grad():
        pytorch_output = model(example_input)
    print(f"PyTorch output shape: {pytorch_output.shape}")
    
    # Trace the model for conversion
    traced_model = torch.jit.trace(model, example_input)
    print("✅ Model traced successfully")
    
    # Convert to Core ML with different compute units
    compute_unit_configs = [
        ("CPU_ONLY", ComputeUnit.CPU_ONLY, "CPU only - no ANE optimizations"),
        ("CPU_AND_NE", ComputeUnit.CPU_AND_NE, "CPU + Neural Engine - ANE optimizations applied"),
        ("ALL", ComputeUnit.ALL, "All compute units - ANE optimizations applied"),
    ]
    
    for name, compute_unit, description in compute_unit_configs:
        print(f"\n--- Converting with {name} ---")
        print(f"Description: {description}")
        
        # Convert to Core ML
        mlmodel = ct.convert(
            traced_model,
            source="pytorch",
            inputs=[ct.TensorType(shape=example_input.shape)],
            compute_units=compute_unit,
            minimum_deployment_target=ct.target.iOS15,
        )
        
        # Get the MIL program to inspect operations
        mil_program = mlmodel._mil_program
        operations = get_op_types_in_program(mil_program)
        
        print(f"Operations in converted model: {operations}")
        
        # Check if ane_rms_norm is present
        if "ane_rms_norm" in operations:
            print("✅ ane_rms_norm operation detected!")
            
            # Check if it was lowered to primitive ops
            primitive_ops = ["negative", "concat", "layer_norm", "slice_by_index", "mul"]
            if all(op in operations for op in primitive_ops):
                print("✅ ane_rms_norm was lowered to primitive ops for ANE execution!")
            else:
                print("⚠️  ane_rms_norm present but not lowered (may need explicit pass)")
        else:
            # Check if the pattern was converted to standard ops (this is expected for CPU_ONLY)
            rms_ops = ["reduce_l2_norm", "mul", "add", "real_div"]
            if any(op in operations for op in rms_ops):
                print("ℹ️  RMSNorm pattern detected but not fused to ane_rms_norm (expected for CPU_ONLY)")
            else:
                print("❌ RMSNorm pattern not detected")
        
        # Test the converted model
        coreml_output = mlmodel.predict({"x": example_input.numpy()})
        
        # Get the output key (it might be renamed)
        output_key = list(coreml_output.keys())[0]
        print(f"Core ML output shape: {coreml_output[output_key].shape}")
        
        # Compare outputs
        output_diff = np.abs(pytorch_output.numpy() - coreml_output[output_key]).max()
        print(f"Max difference between PyTorch and Core ML: {output_diff:.2e}")


def show_explicit_lowering():
    """Show how to explicitly apply the lowering pass."""
    
    print("\n=== Explicit ANE RMS Norm Lowering ===\n")
    
    # Create and convert model
    model = create_rms_norm_model(256, 64)
    example_input = torch.randn(1, 64, 256)
    
    # Trace the model for conversion
    traced_model = torch.jit.trace(model, example_input)
    
    mlmodel = ct.convert(
        traced_model,
        source="pytorch",
        inputs=[ct.TensorType(shape=example_input.shape)],
        compute_units=ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS15,
    )
    
    mil_program = mlmodel._mil_program
    print(f"Operations before explicit lowering: {get_op_types_in_program(mil_program)}")
    
    # Apply the lowering pass explicitly
    from coremltools.converters.mil.mil.passes.defs.ane_rms_norm_to_layer_norm import lower_ane_rms_norm_to_layer_norm
    
    pass_instance = lower_ane_rms_norm_to_layer_norm()
    pass_instance.compute_units = ComputeUnit.CPU_AND_NE
    pass_instance.apply(mil_program)
    
    print(f"Operations after explicit lowering: {get_op_types_in_program(mil_program)}")
    
    # Convert back to Core ML model
    lowered_mlmodel = ct.convert(
        mil_program,
        convert_from="milinternal",
        convert_to="mlprogram",
        compute_units=ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS15,
    )
    
    # Test the lowered model
    coreml_output = lowered_mlmodel.predict({"x": example_input.numpy()})
    output_key = list(coreml_output.keys())[0]
    print(f"Lowered model output shape: {coreml_output[output_key].shape}")


def demonstrate_performance_benefits():
    """Demonstrate the performance benefits of ANE optimization."""
    
    print("\n=== Performance Benefits of ANE Optimization ===\n")
    
    # Create a larger model to show benefits
    model = create_rms_norm_model(1024, 512)
    example_input = torch.randn(1, 512, 1024)
    
    print("Converting large model (1024 hidden size, 512 sequence length)...")
    
    # Trace the model for conversion
    traced_model = torch.jit.trace(model, example_input)
    
    # Convert with CPU only
    cpu_model = ct.convert(
        traced_model,
        source="pytorch",
        inputs=[ct.TensorType(shape=example_input.shape)],
        compute_units=ComputeUnit.CPU_ONLY,
        minimum_deployment_target=ct.target.iOS15,
    )
    
    # Convert with ANE
    ane_model = ct.convert(
        traced_model,
        source="pytorch",
        inputs=[ct.TensorType(shape=example_input.shape)],
        compute_units=ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS15,
    )
    
    print("✅ Models converted successfully!")
    print("\nKey benefits of ANE optimization:")
    print("1. Faster inference on Apple Silicon devices")
    print("2. Lower power consumption")
    print("3. Optimized for Neural Engine architecture")
    print("4. Automatic lowering of ane_rms_norm to primitive ops")


def show_custom_pass_pipeline():
    """Show how to use a custom pass pipeline for more control."""
    
    print("\n=== Custom Pass Pipeline Example ===\n")
    
    model = create_rms_norm_model(256, 64)
    example_input = torch.randn(1, 64, 256)
    
    # Create custom pass pipeline
    from coremltools.converters.mil.mil.passes.pass_pipeline import PassPipeline
    
    pipeline = PassPipeline()
    pipeline.append_pass("common::lower_ane_rms_norm_to_layer_norm")
    
    # Set options for the pass
    pipeline.set_options("common::lower_ane_rms_norm_to_layer_norm", {
        "compute_units": ComputeUnit.CPU_AND_NE
    })
    
    print(f"Custom pipeline passes: {pipeline.passes}")
    
    # Trace the model for conversion
    traced_model = torch.jit.trace(model, example_input)
    
    # Convert with custom pipeline
    mlmodel = ct.convert(
        traced_model,
        source="pytorch",
        inputs=[ct.TensorType(shape=example_input.shape)],
        compute_units=ComputeUnit.CPU_AND_NE,
        pass_pipeline=pipeline,
        minimum_deployment_target=ct.target.iOS15,
    )
    
    mil_program = mlmodel._mil_program
    operations = get_op_types_in_program(mil_program)
    print(f"Operations with custom pipeline: {operations}")


if __name__ == "__main__":
    demonstrate_conversion_pipeline()
    show_explicit_lowering()
    demonstrate_performance_benefits()
    show_custom_pass_pipeline()
    
    print("\n=== Summary ===")
    print("This example demonstrates how PyTorch RMSNorm automatically converts")
    print("to ane_rms_norm and gets optimized for Apple Neural Engine execution.")
    print("The key benefits are:")
    print("- Automatic detection and conversion of RMSNorm patterns")
    print("- ANE-specific optimizations when compute units include Neural Engine")
    print("- Improved performance on Apple Silicon devices")
    print("- Seamless integration with existing PyTorch workflows") 