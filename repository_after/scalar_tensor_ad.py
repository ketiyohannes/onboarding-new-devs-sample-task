
from __future__ import annotations
import math
from typing import Union, Set, Tuple, List, Callable

class Tensor:
    """
    A scalar-value Tensor for automatic differentiation.
    
    This class represents a node in a computational graph, storing both its value (`data`)
    and its local gradient (`grad`). It supports reverse-mode automatic differentiation
    by tracking the operations that created it.
    """
    
    def __init__(self, data: float, _children: Tuple[Tensor, ...] = (), op: str = ""):
        """
        Initialize a Tensor.
        
        Args:
            data: The scalar value of the tensor.
            _children: The parent tensors that produced this tensor (used for backprop).
            op: The operation string (e.g., '+', '*') for debugging/visualization.
        """
        self.data = float(data)
        self.grad = 0.0
        # The backward function computes the gradient of this node's parents.
        # It is a closure that captures the context of the operation.
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set[Tensor] = set(_children)
        self.op = op

    def __repr__(self) -> str:
        return f"Tensor(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other: Union[Tensor, float]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")
        
        def _backward():
            # Gradients distribute equally in addition: d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
            
        out._backward = _backward
        return out

    def __radd__(self, other: Union[Tensor, float]) -> Tensor:
        return self + other

    def __sub__(self, other: Union[Tensor, float]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, (self, other), "-")
        
        def _backward():
            # Gradients flow positively to self and negatively to other
            self.grad += 1.0 * out.grad
            other.grad += -1.0 * out.grad
            
        out._backward = _backward
        return out

    def __rsub__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(other) - self

    def __mul__(self, other: Union[Tensor, float]) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")
        
        def _backward():
            # Product rule: d(a*b)/da = b, d(a*b)/db = a
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
            # Quotient rule derivatives
            self.grad += (1 / other.data) * out.grad
            other.grad -= (self.data / (other.data ** 2)) * out.grad
            
        out._backward = _backward
        return out

    def __rtruediv__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(other) / self

    def __pow__(self, power: Union[int, float]) -> Tensor:
        # only support scalar power (float/int), not Tensor power
        assert isinstance(power, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data ** power, (self,), f"**{power}")
        
        def _backward():
            # Power rule: d(x^n)/dx = n * x^(n-1)
            self.grad += (power * self.data ** (power - 1)) * out.grad
            
        out._backward = _backward
        return out

    def tanh(self) -> Tensor:
        t = math.tanh(self.data)
        out = Tensor(t, (self,), "tanh")
        
        def _backward():
            # d(tanh(x))/dx = 1 - tanh(x)^2
            self.grad += (1 - t ** 2) * out.grad
            
        out._backward = _backward
        return out

    def relu(self) -> Tensor:
        out = Tensor(self.data if self.data > 0 else 0.0, (self,), "relu")
        
        def _backward():
            # ReLU gradient is 1 if data > 0, else 0
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
            
        out._backward = _backward
        return out

    def sigmoid(self) -> Tensor:
        # Sigmoid: 1 / (1 + e^-x)
        s = 1 / (1 + math.exp(-self.data))
        out = Tensor(s, (self,), "sigmoid")
        
        def _backward():
            # d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
            self.grad += s * (1 - s) * out.grad
            
        out._backward = _backward
        return out

    def backward(self) -> None:
        """
        Backpropagates gradient from this node to all ancestors.
        Performs a topological sort to ensure dependencies are processed in order.
        """
        topo: List[Tensor] = []
        visited: Set[Tensor] = set()
        
        def build(v: Tensor):
            if v not in visited:
                visited.add(v)
                
                for child in sorted(v._prev, key=id):
                    build(child)
                topo.append(v)
        
        build(self)
        
        # Initialize gradient of the final node (implicit 1.0 from chain rule: dL/dL = 1)
        self.grad = 1.0
        
        # Apply chain rule in reverse topological order
        for node in reversed(topo):
            node._backward()


if __name__ == "__main__":
    x = Tensor(2.0)
    y = Tensor(3.0)
    z = x * y + x ** 2
    q = z + x
    out = q.relu()
    out.backward()
    print("x:", x.data, "grad:", x.grad)
    print("y:", y.data, "grad:", y.grad)
    print("z:", z.data, "grad:", z.grad)
    print("out:", out.data, "grad:", out.grad)
