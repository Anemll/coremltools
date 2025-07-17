#!/usr/bin/env python3
"""
Example demonstrating how to use the ANE RMS norm pass with compute unit options.

This example shows how to:
1. Create a model with ane_rms_norm operations
2. Apply the lowering pass with different compute unit settings
3. See how the pass behavior changes based on compute units
"""

import numpy as np
import coremltools as ct
from coremltools import ComputeUnit
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil.passes.defs.ane_rms_norm_to_layer_norm import lower_ane_rms_norm_to_layer_norm


def create_model_with_ane_rms_norm():
    """Create a simple model with ane_rms_norm operation."""
    
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 3, 10, 20))])
    def prog(x):
        # Create a learnable weight for RMS normalization
        gamma = np.random.rand(20).astype(np.float32)
        return mb.ane_rms_norm(x=x, gamma=gamma, epsilon=1e-5)
    
    return prog


def demonstrate_compute_units_behavior():
    """Demonstrate how the pass behaves with different compute units."""
    
    print("=== ANE RMS Norm Pass with Compute Units Example ===\n")
    
    # Create the base model
    prog = create_model_with_ane_rms_norm()
    print(f"Original model operations: {prog.get_op_types_in_program()}")
    
    # Test with different compute units
    compute_unit_configs = [
        ("CPU_ONLY", ComputeUnit.CPU_ONLY, "Should skip pass (no ANE)"),
        ("CPU_AND_GPU", ComputeUnit.CPU_AND_GPU, "Should skip pass (no ANE)"),
        ("CPU_AND_NE", ComputeUnit.CPU_AND_NE, "Should run pass (includes ANE)"),
        ("ALL", ComputeUnit.ALL, "Should run pass (includes ANE)"),
    ]
    
    for name, compute_unit, description in compute_unit_configs:
        print(f"\n--- Testing with {name} ---")
        print(f"Description: {description}")
        
        # Create a fresh copy of the program
        prog_copy = create_model_with_ane_rms_norm()
        
        # Create pass instance and set compute units
        pass_instance = lower_ane_rms_norm_to_layer_norm()
        pass_instance.compute_units = compute_unit
        
        # Apply the pass
        pass_instance.apply(prog_copy)
        
        # Check the result
        operations = prog_copy.get_op_types_in_program()
        print(f"Operations after pass: {operations}")
        
        if "ane_rms_norm" in operations:
            print("✅ Pass was SKIPPED (ane_rms_norm still present)")
        else:
            print("✅ Pass was APPLIED (ane_rms_norm replaced with primitive ops)")
    
    print("\n=== Summary ===")
    print("The pass only runs when compute units include the Neural Engine (ANE).")
    print("This ensures that ANE-specific optimizations are only applied when")
    print("the model will actually run on ANE hardware.")


def show_usage_with_pass_pipeline():
    """Show how to use the pass with a custom pass pipeline."""
    
    print("\n=== Using with Pass Pipeline ===")
    
    # Create the model
    prog = create_model_with_ane_rms_norm()
    
    # Create a custom pass pipeline
    from coremltools.converters.mil.mil.passes.pass_pipeline import PassPipeline
    
    # Method 1: Create pipeline and set options
    pipeline = PassPipeline()
    pipeline.append_pass("common::lower_ane_rms_norm_to_layer_norm")
    
    # Set the compute units option
    pipeline.set_options("common::lower_ane_rms_norm_to_layer_norm", {
        "compute_units": ComputeUnit.CPU_AND_NE
    })
    
    print("Pass pipeline created with CPU_AND_NE compute units")
    print(f"Pipeline passes: {pipeline.passes}")
    
    # Method 2: Direct pass instance with options
    print("\nAlternative: Direct pass instance usage")
    pass_instance = lower_ane_rms_norm_to_layer_norm()
    pass_instance.compute_units = ComputeUnit.CPU_AND_NE
    pass_instance.apply(prog)
    
    print(f"Operations after direct pass: {prog.get_op_types_in_program()}")


if __name__ == "__main__":
    demonstrate_compute_units_behavior()
    show_usage_with_pass_pipeline() 