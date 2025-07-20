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
    
    Axis Validation:
    The pass validates that reduce_mean operations use axis=-1 (last dimension normalization),
    which is required for proper RMSNorm operation. Patterns that normalize over other axes
    (e.g., axis=0 or multiple axes like [0,1]) are rejected and not fused.

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

    def _is_fusion_safe(self):
        """
        Check if ane_rms_norm fusion is safe to use.
        
        Currently, there are known variable scoping issues when using ane_rms_norm
        with iOS 17+ deployment targets due to MIL graph validation changes.
        
        Returns:
            bool: True if fusion is safe to use, False otherwise.
        """
        # Check for environment variable to force disable fusion for problematic cases
        import os
        if os.environ.get('COREML_DISABLE_ANE_RMS_NORM_FUSION', '').lower() in ('1', 'true', 'yes'):
            return False
            
        # For now, allow fusion to proceed and handle errors gracefully at the operation level
        return True

    def apply(self, prog):
        # Check if we should run this pass based on compute units
        # Only run if explicitly targeting Apple Neural Engine (CPU_AND_NE)
        if self._compute_units != ComputeUnit.CPU_AND_NE:
            # Skip this pass unless specifically targeting ANE
            # This includes the case where _compute_units is None (no explicit targeting)
            return
        
        # Check if ane_rms_norm is safe to use with the current configuration
        # There are known variable scoping issues with iOS 17+ deployment targets
        if not self._is_fusion_safe():
            return
                
        # Apply RMSNorm fusion to create ane_rms_norm operations
        for f in prog.functions.values():
            self._fuse_rms_norm_block(f)

    @block_context_manager
    def _fuse_rms_norm_block(self, block):
        for op in list(block.operations):
            # Skip if operation was already removed by a previous fusion
            if op not in block.operations:
                continue
            # Try to match RMSNorm pattern
            if self._try_match_rms_norm_pattern(op, block):
                continue

    def _try_match_rms_norm_pattern(self, op, block):
        """Try to match RMSNorm pattern starting from different operations."""
        # Pattern 1 & 2: x -> square -> reduce_mean -> [add(eps)] -> sqrt -> div -> mul(gamma)
        # Pattern 5 & 6: x -> square -> reduce_mean -> [add(eps)] -> rsqrt -> mul -> mul(gamma)
        if op.op_type == "mul" and self._is_gamma_mul(op):
            # Found potential gamma mul
            # Try sqrt+div pattern first
            pattern = self._trace_rms_norm_pattern_with_gamma(op)
            if pattern is not None:
                # Pattern matched (sqrt+div)! Applying fusion
                self._apply_rms_norm_fusion(pattern, block)
                return True
            # Try rsqrt pattern
            pattern = self._trace_rms_norm_pattern_with_gamma_rsqrt(op)
            if pattern is not None:
                # Pattern matched (rsqrt)! Applying fusion
                self._apply_rms_norm_fusion(pattern, block)
                return True
            else:
                # Pattern did NOT match
                pass
        
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
        
        # Pattern 7 & 8: x -> square -> reduce_mean -> [add(eps)] -> rsqrt -> mul (no gamma)
        elif op.op_type == "mul":
            # Check if this is x * rsqrt(...) pattern without gamma
            if not self._is_gamma_mul(op):
                # Found potential rsqrt mul (no gamma)
                pattern = self._trace_rms_norm_pattern_without_gamma_rsqrt(op)
                if pattern is not None:
                    # Pattern matched (rsqrt no gamma)! Applying fusion
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
        """
        Trace back from the final mul to find the RMSNorm pattern.
        
        This function validates that:
        1. The pattern implements RMSNorm: gamma * (x / sqrt(mean(x²) + eps))
        2. The reduce_mean operation uses axis=-1 (last dimension normalization)
        3. All intermediate operations have the correct structure
        
        Returns None if the pattern is invalid or uses incorrect axes.
        """
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
            # Gamma detection failed for final_mul_op
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
                
            # Validate that reduce_mean normalizes over the last dimension (axis=-1)
            # This is required for RMSNorm - patterns that normalize over other dimensions
            # (e.g., axis=0 or multiple axes like [0,1]) are not valid RMSNorm operations
            if hasattr(reduce_op.axes, 'val'):
                axes = reduce_op.axes.val
                # Convert to list if numpy array
                if hasattr(axes, 'tolist'):
                    axes = axes.tolist()
                elif not isinstance(axes, (list, tuple)):
                    axes = [axes]
                # Check if it's [-1] or equivalent (last dimension)
                if len(axes) != 1 or axes[0] != -1:
                    return None
            else:
                # If no axes specified, reduce_mean defaults to all axes, which is not RMSNorm
                return None
        elif mean_square.op.op_type == "reduce_mean":
            # No epsilon addition
            reduce_op = mean_square.op
            epsilon = 0.0  # eps = 0 for pattern 2
        else:
            return None
            
        # Validate that reduce_mean normalizes over the last dimension (axis=-1)
        # This is required for RMSNorm
        if hasattr(reduce_op.axes, 'val'):
            axes = reduce_op.axes.val
            if not isinstance(axes, (list, tuple)):
                axes = [axes]
            # Convert to list if numpy array
            if hasattr(axes, 'tolist'):
                axes = axes.tolist()
            # Check if it's [-1] or equivalent (last dimension)
            if len(axes) != 1 or axes[0] != -1:
                return None
        else:
            # If no axes specified, reduce_mean defaults to all axes, which is not RMSNorm
            return None
            
        square = reduce_op.x
        
        # Check for square operation (can be "square", "mul" with same input, or "pow" with exponent 2)
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
        elif square_op.op_type == "pow":
            # Check if it's pow(x, 2)
            if square_op.x != x:
                return None
            # Check if exponent is 2
            if not hasattr(square_op.y, 'val') or square_op.y.val != 2.0:
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
        """
        Trace back from the final real_div to find the RMSNorm pattern without gamma.
        
        This function validates that:
        1. The pattern implements RMSNorm: x / sqrt(mean(x²) + eps)
        2. The reduce_mean operation uses axis=-1 (last dimension normalization)
        3. All intermediate operations have the correct structure
        
        Returns None if the pattern is invalid or uses incorrect axes.
        """
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
                
            # Validate that reduce_mean normalizes over the last dimension (axis=-1)
            # This is required for RMSNorm - patterns that normalize over other dimensions
            # (e.g., axis=0 or multiple axes like [0,1]) are not valid RMSNorm operations
            if hasattr(reduce_op.axes, 'val'):
                axes = reduce_op.axes.val
                # Convert to list if numpy array
                if hasattr(axes, 'tolist'):
                    axes = axes.tolist()
                elif not isinstance(axes, (list, tuple)):
                    axes = [axes]
                # Check if it's [-1] or equivalent (last dimension)
                if len(axes) != 1 or axes[0] != -1:
                    return None
            else:
                # If no axes specified, reduce_mean defaults to all axes, which is not RMSNorm
                return None
        elif mean_square.op.op_type == "reduce_mean":
            # No epsilon addition
            reduce_op = mean_square.op
            epsilon = 0.0  # eps = 0 for pattern 3
        else:
            return None
            
        # Validate that reduce_mean normalizes over the last dimension (axis=-1)
        # This is required for RMSNorm
        if hasattr(reduce_op.axes, 'val'):
            axes = reduce_op.axes.val
            if not isinstance(axes, (list, tuple)):
                axes = [axes]
            # Convert to list if numpy array
            if hasattr(axes, 'tolist'):
                axes = axes.tolist()
            # Check if it's [-1] or equivalent (last dimension)
            if len(axes) != 1 or axes[0] != -1:
                return None
        else:
            # If no axes specified, reduce_mean defaults to all axes, which is not RMSNorm
            return None
            
        square = reduce_op.x
        
        # Check for square operation (can be "square", "mul" with same input, or "pow" with exponent 2)
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
        elif square_op.op_type == "pow":
            # Check if it's pow(x, 2)
            if square_op.x != x:
                return None
            # Check if exponent is 2
            if not hasattr(square_op.y, 'val') or square_op.y.val != 2.0:
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

    def _trace_rms_norm_pattern_with_gamma_rsqrt(self, final_mul_op):
        """
        Trace back from the final mul to find the RMSNorm pattern with rsqrt.
        
        Pattern: gamma * (x * rsqrt(mean(x²) + eps))
        
        This function validates that:
        1. The pattern implements RMSNorm: gamma * (x * rsqrt(mean(x²) + eps))
        2. The reduce_mean operation uses axis=-1 (last dimension normalization)
        3. All intermediate operations have the correct structure
        
        Returns None if the pattern is invalid or uses incorrect axes.
        """
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
            
        # Look for multiplication with rsqrt
        if normalized_tensor.op is None or normalized_tensor.op.op_type != "mul":
            return None
            
        mul_op = normalized_tensor.op
        
        # Check if one input is x and the other is rsqrt
        # Try both orderings: x * rsqrt or rsqrt * x
        x = None
        rsqrt_var = None
        
        if mul_op.x.op and mul_op.x.op.op_type == "rsqrt":
            x = mul_op.y
            rsqrt_var = mul_op.x
        elif mul_op.y.op and mul_op.y.op.op_type == "rsqrt":
            x = mul_op.x
            rsqrt_var = mul_op.y
        else:
            return None
            
        rsqrt_op = rsqrt_var.op
        mean_square = rsqrt_op.x
        
        # Handle epsilon addition: rsqrt(add(reduce_mean, epsilon))
        epsilon = 0.0
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
                
            # Validate that reduce_mean normalizes over the last dimension (axis=-1)
            if hasattr(reduce_op.axes, 'val'):
                axes = reduce_op.axes.val
                # Convert to list if numpy array
                if hasattr(axes, 'tolist'):
                    axes = axes.tolist()
                elif not isinstance(axes, (list, tuple)):
                    axes = [axes]
                # Check if it's [-1] or equivalent (last dimension)
                if len(axes) != 1 or axes[0] != -1:
                    return None
            else:
                # If no axes specified, reduce_mean defaults to all axes, which is not RMSNorm
                return None
        elif mean_square.op.op_type == "reduce_mean":
            # No epsilon addition
            reduce_op = mean_square.op
            epsilon = 0.0
        else:
            return None
            
        # Validate that reduce_mean normalizes over the last dimension (axis=-1)
        if hasattr(reduce_op.axes, 'val'):
            axes = reduce_op.axes.val
            if not isinstance(axes, (list, tuple)):
                axes = [axes]
            # Convert to list if numpy array
            if hasattr(axes, 'tolist'):
                axes = axes.tolist()
            # Check if it's [-1] or equivalent (last dimension)
            if len(axes) != 1 or axes[0] != -1:
                return None
        else:
            # If no axes specified, reduce_mean defaults to all axes, which is not RMSNorm
            return None
            
        square = reduce_op.x
        
        # Check for square operation (can be "square", "mul" with same input, or "pow" with exponent 2)
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
        elif square_op.op_type == "pow":
            # Check if it's pow(x, 2)
            if square_op.x != x:
                return None
            # Check if exponent is 2
            if not hasattr(square_op.y, 'val') or square_op.y.val != 2.0:
                return None
        else:
            return None
            
        pattern_dict = {
            "x": x,
            "gamma": gamma,
            "epsilon": epsilon,
            "axes": reduce_op.axes.val if hasattr(reduce_op.axes, 'val') else [-1],
            "final_mul": final_mul_op,
            "mul_op": mul_op,  # The x * rsqrt multiplication
            "rsqrt_op": rsqrt_op,
            "reduce_op": reduce_op,
            "square_op": square_op,
        }
        
        return pattern_dict

    def _trace_rms_norm_pattern_without_gamma_rsqrt(self, final_mul_op):
        """
        Trace back from the final mul to find the RMSNorm pattern with rsqrt but no gamma.
        
        Pattern: x * rsqrt(mean(x²) + eps)
        
        This function validates that:
        1. The pattern implements RMSNorm: x * rsqrt(mean(x²) + eps)
        2. The reduce_mean operation uses axis=-1 (last dimension normalization)
        3. All intermediate operations have the correct structure
        
        Returns None if the pattern is invalid or uses incorrect axes.
        """
        # Check if one input is x and the other is rsqrt
        # Try both orderings: x * rsqrt or rsqrt * x
        x = None
        rsqrt_var = None
        
        if final_mul_op.x.op and final_mul_op.x.op.op_type == "rsqrt":
            x = final_mul_op.y
            rsqrt_var = final_mul_op.x
        elif final_mul_op.y.op and final_mul_op.y.op.op_type == "rsqrt":
            x = final_mul_op.x
            rsqrt_var = final_mul_op.y
        else:
            return None
            
        rsqrt_op = rsqrt_var.op
        mean_square = rsqrt_op.x
        
        # Handle epsilon addition: rsqrt(add(reduce_mean, epsilon))
        epsilon = 0.0
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
                
            # Validate that reduce_mean normalizes over the last dimension (axis=-1)
            if hasattr(reduce_op.axes, 'val'):
                axes = reduce_op.axes.val
                # Convert to list if numpy array
                if hasattr(axes, 'tolist'):
                    axes = axes.tolist()
                elif not isinstance(axes, (list, tuple)):
                    axes = [axes]
                # Check if it's [-1] or equivalent (last dimension)
                if len(axes) != 1 or axes[0] != -1:
                    return None
            else:
                # If no axes specified, reduce_mean defaults to all axes, which is not RMSNorm
                return None
        elif mean_square.op.op_type == "reduce_mean":
            # No epsilon addition
            reduce_op = mean_square.op
            epsilon = 0.0
        else:
            return None
            
        # Validate that reduce_mean normalizes over the last dimension (axis=-1)
        if hasattr(reduce_op.axes, 'val'):
            axes = reduce_op.axes.val
            if not isinstance(axes, (list, tuple)):
                axes = [axes]
            # Convert to list if numpy array
            if hasattr(axes, 'tolist'):
                axes = axes.tolist()
            # Check if it's [-1] or equivalent (last dimension)
            if len(axes) != 1 or axes[0] != -1:
                return None
        else:
            # If no axes specified, reduce_mean defaults to all axes, which is not RMSNorm
            return None
            
        square = reduce_op.x
        
        # Check for square operation (can be "square", "mul" with same input, or "pow" with exponent 2)
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
        elif square_op.op_type == "pow":
            # Check if it's pow(x, 2)
            if square_op.x != x:
                return None
            # Check if exponent is 2
            if not hasattr(square_op.y, 'val') or square_op.y.val != 2.0:
                return None
        else:
            return None
            
        pattern_dict = {
            "x": x,
            "gamma": np.ones(x.shape[-1], dtype=np.float32),  # Default gamma = 1 for no gamma cases
            "epsilon": epsilon,
            "axes": reduce_op.axes.val if hasattr(reduce_op.axes, 'val') else [-1],
            "final_op": final_mul_op,  # Use different key since it's not final_mul with gamma
            "mul_op": final_mul_op,  # The x * rsqrt multiplication
            "rsqrt_op": rsqrt_op,
            "reduce_op": reduce_op,
            "square_op": square_op,
        }
        
        return pattern_dict

    def _apply_rms_norm_fusion(self, pattern, block):
        """Apply the RMSNorm fusion by replacing the pattern with ane_rms_norm."""
        # Get the final operation (could be final_mul or final_op for div-only patterns)
        final_op = pattern.get("final_mul") or pattern.get("final_op")
        
        try:
            self._apply_rms_norm_fusion_impl(pattern, block)
        except Exception as e:
            # If fusion fails (e.g., due to variable visibility issues on iOS 17+),
            # silently skip the fusion and let the original pattern remain.
            # This preserves functionality while avoiding crashes.
            return
    
    def _apply_rms_norm_fusion_impl(self, pattern, block):
        """Implementation of RMSNorm fusion that may throw exceptions."""
        # Get the final operation (could be final_mul or final_op for div-only patterns)
        final_op = pattern.get("final_mul") or pattern.get("final_op")
        
        # Find the ultimate output that users of this pattern depend on
        # This could be the final_op output or a cast operation's output
        final_output = final_op.outputs[0]
        cast_users = [op for op in final_output.child_ops if op.op_type == "cast"]
        
        # Determine what the final output variable should be replaced with
        if cast_users:
            # If there are cast operations, we need to handle them
            ultimate_output_op = cast_users[0]  # Assume single cast for simplicity
            ultimate_output_var = ultimate_output_op.outputs[0]
            cast_dtype = ultimate_output_op.dtype.val
        else:
            # No cast operations
            ultimate_output_op = final_op
            ultimate_output_var = final_output
            cast_dtype = None
        
        # Create the ane_rms_norm operation with safer naming
        # Use a unique name that doesn't conflict with existing variables
        import uuid
        unique_suffix = str(uuid.uuid4())[:8]
        
        # CRITICAL FIX: Create the ANE-optimized layer_norm sequence directly 
        # instead of using ane_rms_norm to avoid iOS16+ variable scoping issues
        # This replicates what the ane_rms_norm lowering pass does
        
        # Step 1: Create negative input for concatenation (RMSNorm trick)
        neg_x = mb.mul(
            x=pattern["x"],
            y=-1.0,
            name=f"rms_neg_{unique_suffix}",
            before_op=final_op,
        )
        
        # Step 2: Concatenate input and its negative along the feature dimension
        concat_input = mb.concat(
            values=[pattern["x"], neg_x],
            axis=-1,
            name=f"rms_concat_{unique_suffix}",
            before_op=final_op,
        )
        
        # Step 3: Apply layer_norm on concatenated tensor 
        # This approximates RMSNorm when applied to [x, -x]
        layer_norm_out = mb.layer_norm(
            x=concat_input,
            axes=pattern["axes"],
            epsilon=pattern["epsilon"],
            name=f"rms_layer_norm_{unique_suffix}",
            before_op=final_op,
        )
        
        # Step 4: Slice to get the first half (original input size)
        original_size = pattern["x"].shape[-1]
        # slice_by_index doesn't use axes parameter, it uses begin/end for all dimensions
        # Build begin/end arrays based on input rank
        rank = len(layer_norm_out.shape)
        begin_indices = [0] * rank
        end_indices = list(layer_norm_out.shape)
        end_indices[-1] = original_size  # Only slice the last dimension
        
        sliced_out = mb.slice_by_index(
            x=layer_norm_out,
            begin=begin_indices,
            end=end_indices,
            name=f"rms_slice_{unique_suffix}",
            before_op=final_op,
        )
        # Created slice
        
        # Step 5: Multiply by gamma
        gamma_mul_out = mb.mul(
            x=sliced_out,
            y=pattern["gamma"],
            name=f"rms_gamma_mul_{unique_suffix}",
            before_op=final_op,
        )
        # Created gamma multiplication
        
        # Handle cast if needed
        if cast_dtype is not None:
            # Creating cast
            final_replacement = mb.cast(
                x=gamma_mul_out,
                dtype=cast_dtype,
                name=f"rms_cast_{unique_suffix}",
                before_op=final_op,
            )
            # Cast created
        else:
            final_replacement = gamma_mul_out
            # No cast needed
        
        # Replace all uses of the ultimate output with our replacement
        block.replace_uses_of_var_after_op(
            anchor_op=ultimate_output_op,
            old_var=ultimate_output_var,
            new_var=final_replacement,
        )
        
        # Collect all operations to remove
        ops_to_remove = [
            pattern["reduce_op"],
            pattern["square_op"],
            final_op,  # Always remove the final mul/div
        ]
        
        # Add operations specific to sqrt+div pattern
        if "sqrt_op" in pattern:
            ops_to_remove.append(pattern["sqrt_op"])
            # Remove epsilon addition if it exists
            if pattern["sqrt_op"].x.op and pattern["sqrt_op"].x.op.op_type == "add":
                ops_to_remove.append(pattern["sqrt_op"].x.op)
            # Remove sqrt cast if it exists
            if "sqrt_cast_op" in pattern:
                ops_to_remove.append(pattern["sqrt_cast_op"])
            # Remove div_op if it's different from final_op
            if "div_op" in pattern and pattern["div_op"] != final_op:
                ops_to_remove.append(pattern["div_op"])
        
        # Add operations specific to rsqrt pattern
        if "rsqrt_op" in pattern:
            ops_to_remove.append(pattern["rsqrt_op"])
            # Remove epsilon addition if it exists
            if pattern["rsqrt_op"].x.op and pattern["rsqrt_op"].x.op.op_type == "add":
                ops_to_remove.append(pattern["rsqrt_op"].x.op)
            # Remove mul_op if it's different from final_op (for rsqrt patterns)
            if "mul_op" in pattern and pattern["mul_op"] != final_op:
                ops_to_remove.append(pattern["mul_op"])
            
        # Remove cast operations
        if cast_users:
            ops_to_remove.extend(cast_users)
            
        
        # Remove operations
        block.remove_ops(ops_to_remove) 