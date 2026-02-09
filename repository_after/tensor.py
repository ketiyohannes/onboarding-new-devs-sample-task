import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, _parents=(), _backward=None):
        self.data = np.array(data, dtype=float)
        self.requires_grad = requires_grad
        self.grad = None
        self._parents = _parents
        self._backward = _backward or (lambda: None)
    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
    def _ensure_grad(self):
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

    @staticmethod
    def _unbroadcast(grad, shape):
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for i, (g, s) in enumerate(zip(grad.shape, shape)):
            if s == 1 and g != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:
            if self.data.size != 1:
                raise ValueError("grad must be provided for non-scalar outputs")
            grad = np.ones_like(self.data)
        self._ensure_grad()
        self.grad = self.grad + grad

        topo, seen = [], set()

        def build(v):
            if id(v) in seen:
                return
            seen.add(id(v))
            for p in v._parents:
                build(p)
            topo.append(v)
        build(self)
        for v in reversed(topo):
            v._backward()
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad, (self, other))
        def _bw():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += Tensor._unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other._ensure_grad()
                other.grad += Tensor._unbroadcast(out.grad, other.data.shape)
        out._backward = _bw
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad, (self, other))
        def _bw():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += Tensor._unbroadcast(out.grad * other.data, self.data.shape)
            if other.requires_grad:
                other._ensure_grad()
                other.grad += Tensor._unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _bw
        return out

    def __neg__(self):
        return self * -1.0
    def __sub__(self, other):
        return self + (-other)
    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1)

    def __pow__(self, p):
        out = Tensor(self.data ** p, self.requires_grad, (self,))

        def _bw():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad * (p * (self.data ** (p - 1)))
        out._backward = _bw
        return out

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad, (self, other))

        def _bw():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other._ensure_grad()
                other.grad += self.data.T @ out.grad

        out._backward = _bw
        return out

    def T(self):
        out = Tensor(self.data.T, self.requires_grad, (self,))

        def _bw():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += out.grad.T

        out._backward = _bw
        return out

    def sum(self):
        out = Tensor(self.data.sum(), self.requires_grad, (self,))
        def _bw():
            if self.requires_grad:
                self._ensure_grad()
                self.grad += np.ones_like(self.data) * out.grad
        out._backward = _bw
        return out
    def mean(self):
        return self.sum() * (1.0 / self.data.size)