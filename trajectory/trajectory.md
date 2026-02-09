# Engineering Trajectory: Building a Minimal Autograd Engine

## Analysis: Deconstructing the Problem

When I first looked at this challenge, I needed to build a lightweight automatic differentiation engine from scratch. The core requirements were:

1. **Scalar-based computation graph**: Each Tensor wraps a single float value, not arrays
2. **Reverse-mode autodiff**: Implement backpropagation to compute gradients
3. **Basic operations**: Support arithmetic (+, -, *, /, **) and activations (tanh, relu, sigmoid)
4. **Minimal dependencies**: Only use Python's standard library (math module allowed)
5. **Single file implementation**: Keep it simple and contained

The key insight was recognizing this as a micrograd-style autograd engine - think of it as the computational backbone of neural networks, but stripped down to its essence. Every operation needs to:
- Compute the forward pass (the actual result)
- Store references to parent tensors (build the computation graph)
- Define how gradients flow backward (the chain rule in action)

## Strategy: Why This Architecture?

I chose a **dynamic computational graph** approach for several reasons:

### 1. Graph Construction via Parent Tracking
Each Tensor maintains a `_prev` set containing its parent tensors. This creates an implicit directed acyclic graph (DAG) where:
- Nodes = Tensor objects
- Edges = operations that created them
- The `_op` field labels what operation produced this tensor

This design is elegant because the graph builds itself naturally as you perform operations. No explicit graph construction needed.

### 2. Topological Sort for Backpropagation
The `backward()` method uses a depth-first search to build a topological ordering of the computation graph. This ensures we compute gradients in the correct order - from output back to inputs. 

Why topological sort? Because in a computation like `z = (x * y) + x`, we need to:
1. Compute `dz/dz = 1.0` first
2. Then propagate to the addition node
3. Then to both the multiplication and the direct `x` path
4. Finally accumulate gradients at `x` and `y`

### 3. Closure-Based Gradient Functions
Each operation stores its backward pass as a closure (`_backward`). This captures the local context (parent values, operation type) and defines how gradients flow. For example:

```python
def __mul__(self, other):
    out = Tensor(self.data * other.data, (self, other), "*")
    
    def _backward():
        self.grad += other.data * out.grad  # d(xy)/dx = y
        other.grad += self.data * out.grad  # d(xy)/dy = x
    
    out._backward = _backward
    return out
```

This pattern is clean, composable, and makes the chain rule explicit.

### 4. Gradient Accumulation
Using `+=` instead of `=` for gradients is crucial. When a tensor appears multiple times in a computation (like `x * x + x`), gradients from different paths must sum together. This is the multivariate chain rule in action.

## Execution: Step-by-Step Implementation

### Phase 1: Core Tensor Structure
Started with the foundation:
- `__init__`: Store data as float, initialize grad to 0.0, set up graph tracking
- `__repr__`: Simple string representation for debugging
- `_backward`: Default no-op lambda (leaf nodes don't propagate gradients)

### Phase 2: Backpropagation Engine
Implemented the `backward()` method:
1. Build topological order using DFS with visited set
2. Set output gradient to 1.0 (seed of the chain rule)
3. Traverse in reverse topological order, calling each node's `_backward()`

This is the heart of the engine - everything else just defines local derivatives.

### Phase 3: Arithmetic Operations
Implemented each operation following the same pattern:
- **Addition**: Gradient flows equally to both inputs (derivative is 1)
- **Subtraction**: Positive gradient to left, negative to right
- **Multiplication**: Each input gets the other's value times output gradient
- **Division**: Quotient rule - `d(x/y)/dx = 1/y`, `d(x/y)/dy = -x/y²`
- **Power**: Power rule - `d(x^n)/dx = n*x^(n-1)`

Each operation also handles scalar broadcasting - if you pass a raw number, it wraps it in a Tensor automatically.

### Phase 4: Activation Functions
Added three common neural network activations:

**tanh**: Hyperbolic tangent with derivative `1 - tanh²(x)`
```python
def tanh(self):
    t = math.tanh(self.data)
    out = Tensor(t, (self,), "tanh")
    
    def _backward():
        self.grad += (1 - t**2) * out.grad
    
    out._backward = _backward
    return out
```

**ReLU**: Rectified Linear Unit - pass through if positive, zero otherwise
- Derivative is 1 for positive inputs, 0 for negative
- Simple but powerful for deep learning

**Sigmoid**: Smooth S-curve with derivative `σ(x) * (1 - σ(x))`
- Used in binary classification and gates (LSTM, etc.)

### Phase 5: Testing & Validation
All 15 tests passed, covering:
- Basic tensor properties and graph construction
- Each arithmetic operation with gradient checks
- Each activation function with gradient checks
- Complex expressions with gradient accumulation
- Topological ordering correctness
- Implementation constraints (single file, no external deps)

## Resources & Concepts Used

### Core Concepts
- **Automatic Differentiation**: [Wikipedia - Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
- **Reverse Mode AD (Backpropagation)**: The algorithm that powers neural network training
- **Computational Graphs**: DAG representation of mathematical expressions
- **Chain Rule**: The calculus foundation of backpropagation

### Inspiration & References
- **micrograd** by Andrej Karpathy: The spiritual predecessor - a minimal autograd engine for education
- **PyTorch Autograd**: The production-grade version of these concepts
- **Calculus Derivatives**: Standard derivative rules for each operation

### Python Techniques
- **Closures**: Capturing local scope in `_backward` functions
- **Operator Overloading**: Making Tensors work with `+`, `-`, `*`, etc.
- **DFS & Topological Sort**: Graph traversal algorithms
- **Type Checking**: `isinstance()` for scalar broadcasting

### Mathematical Foundations
- **Multivariate Chain Rule**: Why gradients accumulate with `+=`
- **Partial Derivatives**: Each operation computes ∂output/∂input
- **Gradient Descent**: This engine enables optimization algorithms
