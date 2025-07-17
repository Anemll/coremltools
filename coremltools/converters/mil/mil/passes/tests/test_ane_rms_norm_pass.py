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


if __name__ == "__main__":
    test = TestLowerAneRmsNormToLayerNorm()
    test.test_lower_ane_rms_norm() 