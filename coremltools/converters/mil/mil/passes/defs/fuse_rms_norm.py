#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.graph_pass import AbstractGraphPass
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools import ComputeUnit


@register_pass(namespace="common")
class fuse_rms_norm(AbstractGraphPass):
    """
    Fuse RMSNorm patterns into ane_rms_norm operations.
    
    This pass detects patterns that implement RMSNorm:
        rms = sqrt(mean(x²))
        output = gamma * (x / (rms + eps))
    
    And converts them to ane_rms_norm operations for ANE optimization.
    
    This pass only runs when the target compute unit includes the Neural Engine (ANE).
    Specifically, it requires ``ComputeUnit.CPU_AND_NE`` or ``ComputeUnit.ALL`` to ensure ANE availability.
    
    Supported patterns:
    1. Basic RMSNorm: x -> square -> reduce_mean -> add(eps) -> sqrt -> div -> mul(gamma)
    2. eps=0: RMSNorm: x -> square -> reduce_mean -> sqrt -> div -> mul(gamma)
    3. eps=0, no gamma or gamma=1: RMSNorm: x -> square -> reduce_mean -> sqrt -> div
    4. no gamma or gamma=1: RMSNorm: x -> square -> reduce_mean -> add(eps) -> sqrt -> div 

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
                
        # Apply RMSNorm fusion to create ane_rms_norm operations
        for f in prog.functions.values():
            self._fuse_rms_norm_block(f)

    @block_context_manager
    def _fuse_rms_norm_block(self, block):
        for op in list(block.operations):
            # Try to match RMSNorm pattern
            if self._try_match_rms_norm_pattern(op, block):
                continue

    def _try_match_rms_norm_pattern(self, op, block):
        """Try to match RMSNorm pattern starting from different operations."""
        # Pattern 1 & 2: x -> square -> reduce_mean -> [add(eps)] -> sqrt -> div -> mul(gamma)
        if op.op_type == "mul" and self._is_gamma_mul(op):
            pattern = self._trace_rms_norm_pattern_with_gamma(op)
            if pattern is not None:
                self._apply_rms_norm_fusion(pattern, block)
                return True
        
        # Pattern 3 & 4: x -> square -> reduce_mean -> [add(eps)] -> sqrt -> div (no gamma)
        elif op.op_type == "real_div":
            # Check if this div is NOT followed by a mul(gamma) - only then it's a no-gamma pattern
            div_output = op.outputs[0]
            is_followed_by_gamma_mul = any(
                user.op_type == "mul" and self._is_gamma_mul(user)
                for user in div_output.child_ops
            )
            
            if not is_followed_by_gamma_mul:
                pattern = self._trace_rms_norm_pattern_without_gamma(op)
                if pattern is not None:
                    self._apply_rms_norm_fusion(pattern, block)
                    return True
                
        return False

    def _is_gamma_mul(self, op):
        """Check if this mul operation is multiplying by a learnable weight (gamma)."""
        # Check if either operand is a constant
        if op.x.val is not None:
            # x is a constant (gamma)
            return True
        elif op.y.val is not None:
            # y is a constant (gamma)
            return True
        
        # Check if either operand comes from a cast of a constant (PyTorch pattern)
        if op.x.op and op.x.op.op_type == "cast" and op.x.op.x.val is not None:
            return True
        elif op.y.op and op.y.op.op_type == "cast" and op.y.op.x.val is not None:
            return True
            
        return False

    def _trace_rms_norm_pattern_with_gamma(self, final_mul_op):
        """Trace back from the final mul to find the RMSNorm pattern."""
        # Get the input that's not gamma (the normalized tensor)
        if final_mul_op.x.val is not None:
            normalized_tensor = final_mul_op.y
            gamma = final_mul_op.x.val
        elif final_mul_op.y.val is not None:
            normalized_tensor = final_mul_op.x
            gamma = final_mul_op.y.val
        elif final_mul_op.x.op and final_mul_op.x.op.op_type == "cast" and final_mul_op.x.op.x.val is not None:
            normalized_tensor = final_mul_op.y
            gamma = final_mul_op.x.op.x.val
        elif final_mul_op.y.op and final_mul_op.y.op.op_type == "cast" and final_mul_op.y.op.x.val is not None:
            normalized_tensor = final_mul_op.x
            gamma = final_mul_op.y.op.x.val
        else:
            return None
            
        # Look for division by RMS
        if normalized_tensor.op is None or normalized_tensor.op.op_type != "real_div":
            return None
            
        div_op = normalized_tensor.op
        x = div_op.x
        rms = div_op.y
        
        # Check if rms is sqrt of mean of squares (may be cast(sqrt))
        if rms.op is None:
            return None
            
        # Handle cast(sqrt) pattern
        sqrt_cast_op = None
        if rms.op.op_type == "cast":
            if rms.op.x.op is None or rms.op.x.op.op_type != "sqrt":
                return None
            sqrt_cast_op = rms.op  # Store the cast operation
            sqrt_op = rms.op.x.op
            mean_square = sqrt_op.x
        elif rms.op.op_type == "sqrt":
            sqrt_op = rms.op
            mean_square = sqrt_op.x
        else:
            return None
        
        # Handle epsilon addition: sqrt(add(reduce_mean, epsilon))
        if mean_square.op is None:
            return None
        elif mean_square.op.op_type == "add":
            # This is epsilon addition: reduce_mean + epsilon
            add_op = mean_square.op
            # One input should be reduce_mean, the other should be epsilon constant
            if add_op.x.op and add_op.x.op.op_type == "reduce_mean":
                reduce_op = add_op.x.op
                epsilon = add_op.y.val
            elif add_op.y.op and add_op.y.op.op_type == "reduce_mean":
                reduce_op = add_op.y.op
                epsilon = add_op.x.val
            else:
                return None
        elif mean_square.op.op_type == "reduce_mean":
            # No epsilon addition
            reduce_op = mean_square.op
            epsilon = 0.0  # eps = 0 for pattern 2
        else:
            return None
            
        square = reduce_op.x
        
        # Check for square operation (can be either "square" or "mul" with same input)
        if square.op is None:
            return None
            
        square_op = square.op
        if square_op.op_type == "square":
            if square_op.x != x:
                return None
        elif square_op.op_type == "mul":
            # Check if it's x * x (self-multiplication)
            if square_op.x != square_op.y:
                return None
            
            # Check if either the square inputs match x directly, or x is a cast of the square inputs
            square_input = square_op.x  # Both inputs are the same
            if square_input == x:
                pass  # Direct match
            elif x.op and x.op.op_type == "cast" and x.op.x == square_input:
                pass  # Cast match
            else:
                return None
        else:
            return None
            
        pattern_dict = {
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
        
        # Add cast operations if they exist
        if sqrt_cast_op:
            pattern_dict["sqrt_cast_op"] = sqrt_cast_op
            
        return pattern_dict


    def _trace_rms_norm_pattern_without_gamma(self, final_div_op):
        """Trace back from the final real_div to find the RMSNorm pattern without gamma."""
        # Pattern 3 & 4: x -> square -> reduce_mean -> [add(eps)] -> sqrt -> real_div
        
        # Get the input and RMS from the division
        x = final_div_op.x
        rms = final_div_op.y
        
        # Check if rms is sqrt of mean of squares (may be cast(sqrt))
        if rms.op is None:
            return None
            
        # Handle cast(sqrt) pattern
        sqrt_cast_op = None
        if rms.op.op_type == "cast":
            if rms.op.x.op is None or rms.op.x.op.op_type != "sqrt":
                return None
            sqrt_cast_op = rms.op  # Store the cast operation
            sqrt_op = rms.op.x.op
            mean_square = sqrt_op.x
        elif rms.op.op_type == "sqrt":
            sqrt_op = rms.op
            mean_square = sqrt_op.x
        else:
            return None
        
        # Handle epsilon addition: sqrt(add(reduce_mean, epsilon))
        if mean_square.op is None:
            return None
        elif mean_square.op.op_type == "add":
            # This is epsilon addition: reduce_mean + epsilon
            add_op = mean_square.op
            # One input should be reduce_mean, the other should be epsilon constant
            if add_op.x.op and add_op.x.op.op_type == "reduce_mean":
                reduce_op = add_op.x.op
                epsilon = add_op.y.val
            elif add_op.y.op and add_op.y.op.op_type == "reduce_mean":
                reduce_op = add_op.y.op
                epsilon = add_op.x.val
            else:
                return None
        elif mean_square.op.op_type == "reduce_mean":
            # No epsilon addition
            reduce_op = mean_square.op
            epsilon = 0.0  # eps = 0 for pattern 3
        else:
            return None
            
        square = reduce_op.x
        
        # Check for square operation (can be either "square" or "mul" with same input)
        if square.op is None:
            return None
            
        square_op = square.op
        if square_op.op_type == "square":
            if square_op.x != x:
                return None
        elif square_op.op_type == "mul":
            # Check if it's x * x (self-multiplication)
            if square_op.x != square_op.y:
                return None
            
            # Check if either the square inputs match x directly, or x is a cast of the square inputs
            square_input = square_op.x  # Both inputs are the same
            if square_input == x:
                pass  # Direct match
            elif x.op and x.op.op_type == "cast" and x.op.x == square_input:
                pass  # Cast match
            else:
                return None
        else:
            return None
            
        pattern_dict = {
            "x": x,
            "gamma": np.ones(x.shape[-1], dtype=np.float32),  # Default gamma = 1 for no gamma cases
            "epsilon": epsilon,
            "axes": reduce_op.axes.val if hasattr(reduce_op.axes, 'val') else [-1],
            "final_op": final_div_op,  # Use different key since it's not final_mul
            "div_op": final_div_op,
            "sqrt_op": sqrt_op,
            "reduce_op": reduce_op,
            "square_op": square_op,
        }
        
        # Add cast operations if they exist
        if sqrt_cast_op:
            pattern_dict["sqrt_cast_op"] = sqrt_cast_op
            
        return pattern_dict

    def _apply_rms_norm_fusion(self, pattern, block):
        """Apply the RMSNorm fusion by replacing the pattern with ane_rms_norm."""
        # Get the final operation (could be final_mul or final_op for div-only patterns)
        final_op = pattern.get("final_mul") or pattern.get("final_op")
        
        # Create the ane_rms_norm operation
        ane_rms_norm_out = mb.ane_rms_norm(
            x=pattern["x"],
            gamma=pattern["gamma"],
            epsilon=pattern["epsilon"],
            axes=pattern["axes"],
            name=final_op.name,
            before_op=final_op,
        )
        
        # Replace the final operation output with the ane_rms_norm output
        final_op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=final_op,
            old_var=final_op.outputs[0],
            new_var=ane_rms_norm_out,
        )
        
        # Remove only the core pattern operations
        # Let the cleanup passes handle cast operations
        ops_to_remove = [
            pattern["sqrt_op"],
            pattern["reduce_op"],
            pattern["square_op"],
        ]
        
        # Also remove the add op if it exists (epsilon addition)
        if pattern["sqrt_op"].x.op and pattern["sqrt_op"].x.op.op_type == "add":
            ops_to_remove.append(pattern["sqrt_op"].x.op)
        
        # Remove cast operation for sqrt if it exists
        if "sqrt_cast_op" in pattern:
            ops_to_remove.append(pattern["sqrt_cast_op"])
        
        # Remove div_op but only if it's different from final_op
        if pattern["div_op"] != final_op:
            ops_to_remove.append(pattern["div_op"])
            
        # Remove final_op last to ensure replacement happens first
        ops_to_remove.append(final_op)
            
        # Remove operations
        block.remove_ops(ops_to_remove) 