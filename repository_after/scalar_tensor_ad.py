from __future__ import annotations
import math
from typing import Union, Set, Tuple, List, Callable

class Tensor:
    """A scalar Tensor with automatic differentiation support."""
    
    def __init__(self, data: float, _children: Tuple[Tensor, ...] = (), op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set[Tensor] = set(_children)
        self.op = op

    def __repr__(self) -> str:
        return f"Tensor(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other: Union[Tensor, float]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other: Union[Tensor, float]) -> Tensor:
        return self + other

    def __sub__(self, other: Union[Tensor, float]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, (self, other), "-")
        
        def _backward():
            self.grad += out.grad
            other.grad -= out.grad
        out._backward = _backward
        return out

    def __rsub__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(other) - self

    def __mul__(self, other: Union[Tensor, float]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other: Union[Tensor, float]) -> Tensor:
        return self * other

    def __truediv__(self, other: Union[Tensor, float]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, (self, other), "/")
        
        def _backward():
            self.grad += (1 / other.data) * out.grad
            other.grad -= (self.data / (other.data ** 2)) * out.grad
        out._backward = _backward
        return out

    def __rtruediv__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(other) / self

    def __pow__(self, power: Union[int, float]) -> Tensor:
        assert isinstance(power, (int, float)), "Only int/float powers supported"
        out = Tensor(self.data ** power, (self,), f"**{power}")
        
        def _backward():
            self.grad += (power * self.data ** (power - 1)) * out.grad
        out._backward = _backward
        return out

    def tanh(self) -> Tensor:
        t = math.tanh(self.data)
        out = Tensor(t, (self,), "tanh")
        
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def relu(self) -> Tensor:
        out = Tensor(self.data if self.data > 0 else 0.0, (self,), "relu")
        
        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self) -> Tensor:
        s = 1 / (1 + math.exp(-self.data))
        out = Tensor(s, (self,), "sigmoid")
        
        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward
        return out

    def backward(self) -> None:
        """Computes gradients using reverse-mode autodiff."""
        topo: List[Tensor] = []
        visited: Set[Tensor] = set()
        
        def build(v: Tensor):
            if v not in visited:
                visited.add(v)
                # Sort by id to ensure deterministic order within a run
                for child in sorted(v._prev, key=id):
                    build(child)
                topo.append(v)
        
        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()



if __name__ == "__main__":
    # Example 1: Basic Arithmetic & Gradients
    x = Tensor(2.0)
    y = Tensor(3.0)
    z = x * y + x ** 2  # z = 2*3 + 2^2 = 10
    q = z + x           # q = 10 + 2 = 12
    out = q.relu()      # out = 12
    out.backward()
    
    print("--- Example 1: Basic ---")
    print(f"x: {x.data}, grad: {x.grad}")  # Expected grad: 8.0
    print(f"y: {y.data}, grad: {y.grad}")  # Expected grad: 2.0
    print(f"out: {out.data}\n")

    # Example 2: Activations and Chains
    a = Tensor(0.5)
    b = a.tanh()
    c = b.sigmoid()
    c.backward()
    
    print("--- Example 2: Activations ---")
    print(f"a: {a.data}, grad: {a.grad:.4f}")
    print(f"c: {c.data:.4f}\n")

    # Example 3: Gradient Accumulation (Diamond Graph)
    # top -> left=top*2, right=top*3 -> bottom=left+right
    top = Tensor(2.0)
    left = top * 2.0
    right = top * 3.0
    bottom = left + right
    bottom.backward()
    
    print("--- Example 3: Accumulation ---")
    print(f"top: {top.data}, grad: {top.grad}")  # Expected grad: 5.0
