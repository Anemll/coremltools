#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import pytest

from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.testing_utils import (
    apply_pass_and_basic_check,
    assert_model_is_valid,
    get_op_types_in_program,
)


class TestFuseRMSNorm:
    def test_fuse_rms_norm_basic(self):
        """Test basic RMSNorm pattern fusion."""
        shape = (1, 3, 10, 20)
        gamma_shape = (20,)

        @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
        def prog(x):
            # Create RMSNorm pattern: x -> square -> reduce_mean -> add(eps) -> sqrt -> div -> mul(gamma)
            gamma = np.random.rand(*gamma_shape).astype(np.float32)
            epsilon = 1e-5
            
            # Square the input
            x_squared = mb.square(x=x, name="square")
            
            # Reduce mean over the last dimension
            mean_squared = mb.reduce_mean(x=x_squared, axes=[-1], keep_dims=True, name="reduce_mean")
            
            # Add epsilon
            mean_squared_eps = mb.add(x=mean_squared, y=epsilon, name="add_eps")
            
            # Square root
            rms = mb.sqrt(x=mean_squared_eps, name="sqrt")
            
            # Divide input by RMS
            normalized = mb.real_div(x=x, y=rms, name="div")
            
            # Multiply by gamma
            output = mb.mul(x=gamma, y=normalized, name="mul_gamma")
            
            return output

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_rms_norm"
        )
        
        # Check that the pattern was fused
        assert "ane_rms_norm" in get_op_types_in_program(prog)
        
        # Check that the original ops were removed
        original_ops = ["square", "reduce_mean", "add_eps", "sqrt", "div", "mul_gamma"]
        for op_type in original_ops:
            assert op_type not in get_op_types_in_program(prog)

        # Verify the output numerically
        def np_rms_norm(x, gamma, epsilon=1e-5):
            mean_square = np.mean(np.square(x), axis=-1, keepdims=True)
            normalized_x = x * (1.0 / np.sqrt(mean_square + epsilon))
            return normalized_x * gamma

        input_data = np.random.rand(*shape).astype(np.float32)
        gamma_val = prog.find_ops(op_type="ane_rms_norm")[0].gamma.val
        expected_output = np_rms_norm(input_data, gamma_val)

        output = prog.find_ops(op_type="ane_rms_norm")[0].outputs[0]
        assert_model_is_valid(
            prog,
            {"x": input_data},
            expected_output_shapes={output.name: shape},
            expected_outputs={output.name: expected_output},
        )

    def test_fuse_rms_norm_without_epsilon_add(self):
        """Test RMSNorm pattern without explicit epsilon addition."""
        shape = (1, 3, 10, 20)
        gamma_shape = (20,)

        @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
        def prog(x):
            # Create RMSNorm pattern without explicit epsilon addition
            gamma = np.random.rand(*gamma_shape).astype(np.float32)
            
            # Square the input
            x_squared = mb.square(x=x, name="square")
            
            # Reduce mean over the last dimension
            mean_squared = mb.reduce_mean(x=x_squared, axes=[-1], keep_dims=True, name="reduce_mean")
            
            # Square root (no epsilon addition)
            rms = mb.sqrt(x=mean_squared, name="sqrt")
            
            # Divide input by RMS
            normalized = mb.real_div(x=x, y=rms, name="div")
            
            # Multiply by gamma
            output = mb.mul(x=gamma, y=normalized, name="mul_gamma")
            
            return output

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_rms_norm"
        )
        
        # Check that the pattern was fused
        assert "ane_rms_norm" in get_op_types_in_program(prog)

    def test_fuse_rms_norm_different_axes(self):
        """Test RMSNorm pattern with different reduction axes."""
        shape = (1, 3, 10, 20)
        gamma_shape = (10, 20)

        @mb.program(input_specs=[mb.TensorSpec(shape=shape)])
        def prog(x):
            # Create RMSNorm pattern with reduction over last two dimensions
            gamma = np.random.rand(*gamma_shape).astype(np.float32)
            epsilon = 1e-5
            
            # Square the input
            x_squared = mb.square(x=x, name="square")
            
            # Reduce mean over the last two dimensions
            mean_squared = mb.reduce_mean(x=x_squared, axes=[-2, -1], keep_dims=True, name="reduce_mean")
            
            # Add epsilon
            mean_squared_eps = mb.add(x=mean_squared, y=epsilon, name="add_eps")
            
            # Square root
            rms = mb.sqrt(x=mean_squared_eps, name="sqrt")
            
            # Divide input by RMS
            normalized = mb.real_div(x=x, y=rms, name="div")
            
            # Multiply by gamma
            output = mb.mul(x=gamma, y=normalized, name="mul_gamma")
            
            return output

        prev_prog, prev_block, block = apply_pass_and_basic_check(
            prog, "common::fuse_rms_norm"
        )
        
        # Check that the pattern was fused
        assert "ane_rms_norm" in get_op_types_in_program(prog)
        
        # Check that the axes are preserved
        ane_rms_norm_op = prog.find_ops(op_type="ane_rms_norm")[0]
        assert ane_rms_norm_op.axes.val == [-2, -1] 