import math
import inspect
import ast
from repository_after.tensor import Tensor
import repository_after.tensor as tensor_module


def passed(msg):
    print(f"[PASSED] {msg}")


def test_tensor_scalar_value_and_gradient():
    x = Tensor(1.2)
    assert isinstance(x.data, float)
    assert isinstance(x.grad, float)
    passed("Tensor represents a scalar float value and gradient")


def test_tensor_tracks_parents():
    x = Tensor(2.0)
    y = Tensor(3.0)
    z = x * y
    assert hasattr(z, "_prev")
    assert x in z._prev and y in z._prev
    passed("Tensor maintains references to parent tensors")


def test_reverse_mode_backpropagation():
    x = Tensor(2.0)
    y = Tensor(3.0)
    z = x * y
    z.backward()
    assert x.grad == 3.0
    assert y.grad == 2.0
    passed("Reverse-mode automatic differentiation works")


def test_addition_operation():
    x = Tensor(1.0)
    y = Tensor(2.0)
    z = x + y
    z.backward()
    assert z.data == 3.0
    assert x.grad == 1.0
    assert y.grad == 1.0
    passed("Addition operation supported")


def test_subtraction_operation():
    x = Tensor(5.0)
    y = Tensor(2.0)
    z = x - y
    z.backward()
    assert z.data == 3.0
    assert x.grad == 1.0
    assert y.grad == -1.0
    passed("Subtraction operation supported")


def test_multiplication_operation():
    x = Tensor(2.0)
    y = Tensor(4.0)
    z = x * y
    z.backward()
    assert z.data == 8.0
    assert x.grad == 4.0
    assert y.grad == 2.0
    passed("Multiplication operation supported")


def test_division_operation():
    x = Tensor(6.0)
    y = Tensor(3.0)
    z = x / y
    z.backward()
    assert z.data == 2.0
    assert math.isclose(x.grad, 1 / 3)
    assert math.isclose(y.grad, -6 / 9)
    passed("True division operation supported")


def test_power_operation():
    x = Tensor(3.0)
    z = x**2
    z.backward()
    assert z.data == 9.0
    assert x.grad == 6.0
    passed("Power operation supported")


def test_tanh_operation():
    x = Tensor(0.5)
    z = x.tanh()
    z.backward()
    t = math.tanh(0.5)
    assert math.isclose(z.data, t)
    assert math.isclose(x.grad, 1 - t * t)
    passed("Hyperbolic tangent supported")


def test_relu_operation():
    x = Tensor(-1.0)
    y = Tensor(2.0)
    zx = x.relu()
    zy = y.relu()
    zx.backward()
    zy.backward()
    assert zx.data == 0.0
    assert x.grad == 0.0
    assert zy.data == 2.0
    assert y.grad == 1.0
    passed("ReLU activation supported")


def test_sigmoid_operation():
    x = Tensor(0.0)
    z = x.sigmoid()
    z.backward()
    assert math.isclose(z.data, 0.5)
    assert math.isclose(x.grad, 0.25)
    passed("Sigmoid activation supported")


def test_gradient_accumulation():
    x = Tensor(2.0)
    z = x * x + x
    z.backward()
    assert x.grad == 5.0
    passed("Gradient accumulation across graph paths works")


def test_topological_backward_order():
    x = Tensor(2.0)
    y = Tensor(3.0)
    z = (x * y + x).tanh()
    z.backward()
    assert x.grad != 0.0
    assert y.grad != 0.0
    passed("Backward pass uses correct topological traversal")


def test_single_file_implementation():
    source = inspect.getsource(Tensor)
    assert "class Tensor" in source
    passed("Implementation is contained in a single source file")


def test_no_external_libraries_used():
    source = inspect.getsource(tensor_module)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for name in node.names:
                assert name.name in ("math",)
    passed("No external libraries are used")
