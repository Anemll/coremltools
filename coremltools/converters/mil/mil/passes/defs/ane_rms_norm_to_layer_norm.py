#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.passes.helper import block_context_manager


@register_pass(namespace="common")
class lower_ane_rms_norm_to_layer_norm(AbstractGraphPass):
    """
    Lower ane_rms_norm to a sequence of more primitive ops that can be executed on ANE.
    This pass replaces ane_rms_norm with the following sequence:
        1. Negative of the input
        2. Concatenate input and its negative
        3. Layer Normalization on the concatenated tensor
        4. Slice to get the first half
        5. Multiply by the learnable weight (gamma)
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._lower_ane_rms_norm_to_layer_norm_block(f)

    @block_context_manager
    def _lower_ane_rms_norm_to_layer_norm_block(self, block):
        for op in list(block.operations):
            if op.op_type == "ane_rms_norm":
                self._replace_op(op)

    def _replace_op(self, op):
        # 1. Negative of the input
        neg_x = mb.negative(x=op.x, name=op.name + "_neg")

        # 2. Concatenate input and its negative
        # The last axis is the feature dimension for RMSNorm
        axis = op.axes.val[-1]
        doubled_x = mb.concat(
            values=[op.x, neg_x], axis=axis, name=op.name + "_concat"
        )

        # 3. Layer Normalization on the concatenated tensor
        # The normalized shape should be the shape of the last dimension
        normalized_shape = [doubled_x.shape[axis]]
        layer_norm_out = mb.layer_norm(
            x=doubled_x,
            axes=[axis],
            gamma=None,
            beta=None,
            epsilon=op.epsilon,
            name=op.name + "_layer_norm",
        )

        # 4. Slice to get the first half
        slice_out = mb.slice_by_index(
            x=layer_norm_out,
            begin=[0] * op.x.rank,
            end=op.x.shape,
            name=op.name + "_slice",
        )

        # 5. Multiply by the learnable weight (gamma)
        if op.gamma is not None:
            final_out = mb.mul(
                x=slice_out, y=op.gamma, name=op.name + "_mul_gamma"
            )
        else:
            final_out = slice_out

        op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=op, old_var=op.outputs[0], new_var=final_out
        )
        # Remove the original op
        op.enclosing_block.remove_ops([op]) 