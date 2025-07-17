#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import pytest

from coremltools import ComputeUnit
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.testing_utils import (
    apply_pass_and_basic_check,
    assert_model_is_valid,
    get_op_types_in_program,
)
from coremltools.converters.mil.mil.passes.defs.ane_rms_norm_to_layer_norm import lower_ane_rms_norm_to_layer_norm


class TestLowerAneRmsNormToLayerNorm:
    def test_lower_ane_rms_norm(self):
        shape = (1, 3, 10, 20)
        gamma_shape = (20,)

        @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
        def prog(x):
            gamma = np.random.rand(*gamma_shape).astype(np.float32)
            return mb.ane_rms_norm(x=x, gamma=gamma)

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::lower_ane_rms_norm_to_layer_norm"
        )
        assert get_op_types_in_program(prev_prog) == ["ane_rms_norm"]
        assert get_op_types_in_program(prog) == [
            "negative",
            "concat",
            "layer_norm",
            "slice_by_index",
            "mul",
        ]

        # Verify the output numerically
        def np_rms_norm(x, gamma, epsilon=1e-5):
            mean_square = np.mean(np.square(x), axis=-1, keepdims=True)
            normalized_x = x * (1.0 / np.sqrt(mean_square + epsilon))
            return normalized_x * gamma

        input_data = np.random.rand(*shape).astype(np.float32)
        gamma_val = prog.find_ops(op_type="mul")[0].y.val
        expected_output = np_rms_norm(input_data, gamma_val)

        output = prog.find_ops(op_type="mul")[0].outputs[0]
        assert_model_is_valid(
            prog,
            {"x": input_data},
            expected_output_shapes={output.name: shape},
            expected_outputs={output.name: expected_output},
        )

    def test_compute_units_option(self):
        """Test that the pass respects the compute_units option."""
        shape = (1, 3, 10, 20)
        gamma_shape = (20,)

        @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
        def prog(x):
            gamma = np.random.rand(*gamma_shape).astype(np.float32)
            return mb.ane_rms_norm(x=x, gamma=gamma)

        # Test 1: With CPU_ONLY - pass should be skipped
        pass_instance = lower_ane_rms_norm_to_layer_norm()
        pass_instance.compute_units = ComputeUnit.CPU_ONLY
        
        # Apply the pass
        pass_instance.apply(prog)
        
        # The pass should be skipped, so ane_rms_norm should still be there
        assert get_op_types_in_program(prog) == ["ane_rms_norm"]

        # Test 2: With CPU_AND_NE - pass should run
        @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
        def prog2(x):
            gamma = np.random.rand(*gamma_shape).astype(np.float32)
            return mb.ane_rms_norm(x=x, gamma=gamma)

        pass_instance2 = lower_ane_rms_norm_to_layer_norm()
        pass_instance2.compute_units = ComputeUnit.CPU_AND_NE
        
        # Apply the pass
        pass_instance2.apply(prog2)
        
        # The pass should run, so ane_rms_norm should be replaced
        assert get_op_types_in_program(prog2) == [
            "negative",
            "concat",
            "layer_norm",
            "slice_by_index",
            "mul",
        ]

        # Test 3: With ALL - pass should run
        @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
        def prog3(x):
            gamma = np.random.rand(*gamma_shape).astype(np.float32)
            return mb.ane_rms_norm(x=x, gamma=gamma)

        pass_instance3 = lower_ane_rms_norm_to_layer_norm()
        pass_instance3.compute_units = ComputeUnit.ALL
        
        # Apply the pass
        pass_instance3.apply(prog3)
        
        # The pass should run, so ane_rms_norm should be replaced
        assert get_op_types_in_program(prog3) == [
            "negative",
            "concat",
            "layer_norm",
            "slice_by_index",
            "mul",
        ]

        # Test 4: With CPU_AND_GPU - pass should be skipped
        @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
        def prog4(x):
            gamma = np.random.rand(*gamma_shape).astype(np.float32)
            return mb.ane_rms_norm(x=x, gamma=gamma)

        pass_instance4 = lower_ane_rms_norm_to_layer_norm()
        pass_instance4.compute_units = ComputeUnit.CPU_AND_GPU
        
        # Apply the pass
        pass_instance4.apply(prog4)
        
        # The pass should be skipped, so ane_rms_norm should still be there
        assert get_op_types_in_program(prog4) == ["ane_rms_norm"]


if __name__ == "__main__":
    test = TestLowerAneRmsNormToLayerNorm()
    test.test_lower_ane_rms_norm()
    test.test_compute_units_option() 