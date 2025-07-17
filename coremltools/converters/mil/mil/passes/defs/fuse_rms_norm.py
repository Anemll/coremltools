#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.passes.helper import block_context_manager


@register_pass(namespace="common")
class fuse_rms_norm(AbstractGraphPass):
    """
    Fuse RMSNorm patterns into ane_rms_norm operations.
    
    This pass detects patterns that implement RMSNorm:
        rms = sqrt(mean(x²))
        output = gamma * (x / (rms + eps))
    
    And converts them to ane_rms_norm operations for ANE optimization.
    
    Supported patterns:
    1. Basic RMSNorm: x -> square -> reduce_mean -> add(eps) -> sqrt -> div -> mul(gamma)
    2. Alternative: x -> norm -> div -> mul(gamma)
    """

    def apply(self, prog):
        for f in prog.functions.values():
            self._fuse_rms_norm_block(f)

    @block_context_manager
    def _fuse_rms_norm_block(self, block):
        for op in list(block.operations):
            # Try to match RMSNorm pattern
            if self._try_match_rms_norm_pattern(op, block):
                continue

    def _try_match_rms_norm_pattern(self, op, block):
        """Try to match RMSNorm pattern starting from a mul operation."""
        if op.op_type != "mul":
            return False
            
        # Check if this is the final mul with gamma
        if not self._is_gamma_mul(op):
            return False
            
        # Try to trace back to find the RMSNorm pattern
        pattern = self._trace_rms_norm_pattern(op)
        if pattern is None:
            return False
            
        # Apply the transformation
        self._apply_rms_norm_fusion(pattern, block)
        return True

    def _is_gamma_mul(self, op):
        """Check if this mul operation is multiplying by a learnable weight (gamma)."""
        if op.x.val is not None:
            # x is a constant (gamma)
            return True
        elif op.y.val is not None:
            # y is a constant (gamma)
            return True
        return False

    def _trace_rms_norm_pattern(self, final_mul_op):
        """Trace back from the final mul to find the RMSNorm pattern."""
        # Get the input that's not gamma (the normalized tensor)
        if final_mul_op.x.val is not None:
            normalized_tensor = final_mul_op.y
            gamma = final_mul_op.x.val
        else:
            normalized_tensor = final_mul_op.x
            gamma = final_mul_op.y.val
            
        # Look for division by RMS
        if normalized_tensor.op is None or normalized_tensor.op.op_type != "real_div":
            return None
            
        div_op = normalized_tensor.op
        x = div_op.x
        rms = div_op.y
        
        # Check if rms is sqrt of mean of squares
        if rms.op is None or rms.op.op_type != "sqrt":
            return None
            
        sqrt_op = rms.op
        mean_square = sqrt_op.x
        
        if mean_square.op is None or mean_square.op.op_type != "reduce_mean":
            return None
            
        reduce_op = mean_square.op
        square = reduce_op.x
        
        if square.op is None or square.op.op_type != "square":
            return None
            
        square_op = square.op
        if square_op.x != x:
            return None
            
        # Check for epsilon addition
        if mean_square.op_type == "add":
            add_op = mean_square.op
            if add_op.x == reduce_op.outputs[0]:
                epsilon = add_op.y.val
            elif add_op.y == reduce_op.outputs[0]:
                epsilon = add_op.x.val
            else:
                epsilon = 1e-5  # default
        else:
            epsilon = 1e-5  # default
            
        return {
            "x": x,
            "gamma": gamma,
            "epsilon": epsilon,
            "axes": reduce_op.axes.val if hasattr(reduce_op.axes, 'val') else [-1],
            "final_mul": final_mul_op,
            "div_op": div_op,
            "sqrt_op": sqrt_op,
            "reduce_op": reduce_op,
            "square_op": square_op,
        }

    def _apply_rms_norm_fusion(self, pattern, block):
        """Apply the RMSNorm fusion by replacing the pattern with ane_rms_norm."""
        # Create the ane_rms_norm operation
        ane_rms_norm_out = mb.ane_rms_norm(
            x=pattern["x"],
            gamma=pattern["gamma"],
            epsilon=pattern["epsilon"],
            axes=pattern["axes"],
            name=pattern["final_mul"].name,
            before_op=pattern["final_mul"],
        )
        
        # Replace the final mul output with the ane_rms_norm output
        pattern["final_mul"].enclosing_block.replace_uses_of_var_after_op(
            anchor_op=pattern["final_mul"],
            old_var=pattern["final_mul"].outputs[0],
            new_var=ane_rms_norm_out,
        )
        
        # Remove all the ops in the pattern
        ops_to_remove = [
            pattern["final_mul"],
            pattern["div_op"],
            pattern["sqrt_op"],
            pattern["reduce_op"],
            pattern["square_op"],
        ]
        
        # Also remove the add op if it exists
        if hasattr(pattern["reduce_op"], 'x') and pattern["reduce_op"].x.op and pattern["reduce_op"].x.op.op_type == "add":
            ops_to_remove.append(pattern["reduce_op"].x.op)
            
        block.remove_ops(ops_to_remove) 