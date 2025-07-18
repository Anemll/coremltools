#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools import ComputeUnit


@register_pass(namespace="common")
class lower_ane_rms_norm_to_layer_norm(AbstractGraphPass):
    """
    Lower ``ane_rms_norm`` to a sequence of more primitive ops that can be executed on ANE.
    
    This pass replaces ``ane_rms_norm`` with the following sequence:
    
    1. Negative of the input (multiply by -1)
    2. Concatenate input and its negative along the feature dimension
    3. Layer Normalization on the concatenated tensor
    4. Slice to get the first half (original input size)
    5. Multiply by the learnable weight (gamma)
    
    This transformation leverages the property that LayerNorm on concatenated [x, -x] 
    approximates RMSNorm on x, enabling ANE execution while maintaining numerical accuracy.
    
    This pass only runs when the target compute unit includes the Neural Engine (ANE).
    Specifically, it requires ``ComputeUnit.CPU_AND_NE`` to ensure ANE availability.
    
    Parameters
    ----------
    compute_units : ComputeUnit, optional
        Only run this pass if the compute units include Neural Engine.
        Valid values: ``ComputeUnit.ALL``, ``ComputeUnit.CPU_AND_NE``, 
        ``ComputeUnit.CPU_ONLY``, ``ComputeUnit.CPU_AND_GPU``.
        Default: ``None`` (run for all compute units, no filtering).
        
    Notes
    -----
    The ANE RMS norm optimization provides significant performance improvements on
    Apple Neural Engine while maintaining numerical accuracy within acceptable bounds.
    The transformation introduces small approximation errors that are typically
    negligible for most machine learning applications.
    
    Examples
    --------
    >>> from coremltools.converters.mil.mil.passes.defs.ane_rms_norm_to_layer_norm import lower_ane_rms_norm_to_layer_norm
    >>> from coremltools import ComputeUnit
    >>> 
    >>> # Create and configure the pass
    >>> pass_instance = lower_ane_rms_norm_to_layer_norm()
    >>> pass_instance.compute_units = ComputeUnit.CPU_AND_NE
    >>> 
    >>> # Apply to a MIL program
    >>> pass_instance.apply(mil_program)
    """

    def __init__(self):
        self._compute_units = None

    @property
    def compute_units(self):
        return self._compute_units

    @compute_units.setter
    def compute_units(self, compute_units):
        if compute_units is not None and not isinstance(compute_units, ComputeUnit):
            raise TypeError(
                f"compute_units must be of type ComputeUnit, but got {type(compute_units)}"
            )
        self._compute_units = compute_units

    def apply(self, prog):
        # Check if we should run this pass based on compute units
        # Only run if explicitly targeting Apple Neural Engine (CPU_AND_NE)
        if self._compute_units != ComputeUnit.CPU_AND_NE:
            # Skip this pass unless specifically targeting ANE
            # This includes the case where _compute_units is None (no explicit targeting)
            return
        
        for f in prog.functions.values():
            self._lower_ane_rms_norm_to_layer_norm_block(f)

    @block_context_manager
    def _lower_ane_rms_norm_to_layer_norm_block(self, block):
        for op in list(block.operations):
            if op.op_type == "ane_rms_norm":
                self._replace_op(op)

    def _replace_op(self, op):
        # 1. Negative of the input (multiply by -1)
        neg_one = mb.const(val=-1.0, name=op.name + "_neg_one")
        neg_x = mb.mul(x=op.x, y=neg_one, name=op.name + "_neg")

        # 2. Concatenate input and its negative
        # The last axis is the feature dimension for RMSNorm
        axis = op.axes.val[-1]
        doubled_x = mb.concat(
            values=[op.x, neg_x], axis=axis, name=op.name + "_concat"
        )

        # 3. Layer Normalization on the concatenated tensor
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