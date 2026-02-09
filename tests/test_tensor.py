from repository_after.tensor import Tensor
import math


def test_scalar_storage():
    x = Tensor(3.5)
    assert isinstance(x.data, float)
    assert isinstance(x.grad, float)
    assert x.grad == 0.0
    print("test_scalar_storage passed")


def test_addition():
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = a + b
    c.backward()
    assert c.data == 5.0
    assert a.grad == 1.0
    assert b.grad == 1.0
    print("test_addition passed")


def test_subtraction():
    a = Tensor(5.0)
    b = Tensor(2.0)
    c = a - b
    c.backward()
    assert c.data == 3.0
    assert a.grad == 1.0
    assert b.grad == -1.0
    print("test_subtraction passed")


def test_multiplication():
    a = Tensor(2.0)
    b = Tensor(4.0)
    c = a * b
    c.backward()
    assert c.data == 8.0
    assert a.grad == 4.0
    assert b.grad == 2.0
    print("test_multiplication passed")


def test_division():
    a = Tensor(6.0)
    b = Tensor(3.0)
    c = a / b
    c.backward()
    assert c.data == 2.0
    assert abs(a.grad - (1.0 / 3.0)) < 1e-6
    assert abs(b.grad - (-6.0 / 9.0)) < 1e-6
    print("test_division passed")


def test_power():
    a = Tensor(3.0)
    c = a ** 2
    c.backward()
    assert c.data == 9.0
    assert a.grad == 6.0
    print("test_power passed")


def test_chain_rule():
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = a * b + a
    c.backward()
    assert c.data == 8.0
    assert a.grad == 4.0
    assert b.grad == 2.0
    print("test_chain_rule passed")


def test_gradient_accumulation():
    a = Tensor(2.0)
    b = a * a
    c = b + a
    c.backward()
    assert a.grad == 5.0
    print("test_gradient_accumulation passed")


def test_tanh():
    x = Tensor(0.5)
    y = x.tanh()
    y.backward()
    expected = math.tanh(0.5)
    assert abs(y.data - expected) < 1e-6
    assert abs(x.grad - (1 - expected ** 2)) < 1e-6
    print("test_tanh passed")


def test_relu_positive():
    x = Tensor(2.0)
    y = x.relu()
    y.backward()
    assert y.data == 2.0
    assert x.grad == 1.0
    print("test_relu_positive passed")


def test_relu_negative():
    x = Tensor(-1.0)
    y = x.relu()
    y.backward()
    assert y.data == 0.0
    assert x.grad == 0.0
    print("test_relu_negative passed")


def test_sigmoid():
    x = Tensor(0.0)
    y = x.sigmoid()
    y.backward()
    assert abs(y.data - 0.5) < 1e-6
    assert abs(x.grad - 0.25) < 1e-6
    print("test_sigmoid passed")


def test_deep_graph():
    a = Tensor(1.5)
    b = Tensor(2.0)
    c = Tensor(3.0)
    y = ((a * b) + c).tanh() * a
    y.backward()
    assert isinstance(a.grad, float)
    assert isinstance(b.grad, float)
    assert isinstance(c.grad, float)
    print("test_deep_graph passed")


def test_multiple_paths():
    a = Tensor(2.0)
    b = a * a
    c = a + b
    d = c * a
    d.backward()
    assert a.grad == 16.0
    print("test_multiple_paths passed")


def test_backward_on_intermediate():
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = a * b
    d = c + b
    d.backward()
    assert a.grad == 3.0
    assert b.grad == 3.0
    print("test_backward_on_intermediate passed")


if __name__ == "__main__":
    test_scalar_storage()
    test_addition()
    test_subtraction()
    test_multiplication()
    test_division()
    test_power()
    test_chain_rule()
    test_gradient_accumulation()
    test_tanh()
    test_relu_positive()
    test_relu_negative()
    test_sigmoid()
    test_deep_graph()
    test_multiple_paths()
    test_backward_on_intermediate()
    print("All test are passed")
