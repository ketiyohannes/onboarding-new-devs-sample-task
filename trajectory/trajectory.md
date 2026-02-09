# Implementation Trajectory

## Analysis

Build a minimal automatic differentiation engine with:
- Scalar-valued tensors
- Computational graph construction
- Reverse-mode automatic differentiation
- Operations: +, -, *, /, **
- Activation functions: tanh, ReLU, sigmoid
- Python standard library only

## Strategy

Standard autograd pattern:
1. Tensor stores value, gradient, and parent tensors
2. Operations create new tensors and record relationships
3. Each operation defines `_backward()` for local gradients
4. `backward()` performs topological sort and propagates gradients

## Implementation

- Addition: gradient flows equally to both operands
- Multiplication: gradient scaled by other operand
- Power: derivative formula `n * x^(n-1)`
- Division: multiplication with power(-1)
- Activation functions: standard derivative formulas
- Gradient accumulation: uses `+=` for multiple paths
