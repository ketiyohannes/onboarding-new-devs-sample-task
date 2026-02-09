import numpy as np
import pytest

from repository_after.tensor import Tensor
import repository_after.tensor as tensor_module

def close(a, b, atol=1e-7):
    return np.allclose(a, b, atol=atol, rtol=0)


def test_scalar_add_mul_backward():
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)
    z = x * y + x
    z.backward()
    assert close(z.data, 2 * 3 + 2)
    assert close(x.grad, 4.0)  
    assert close(y.grad, 2.0)  


def test_chain_rule_with_power():
    x = Tensor(3.0, requires_grad=True)
    y = (x ** 2) * x  
    y.backward()
    assert close(y.data, 27.0)
    assert close(x.grad, 27.0)  


def test_sum_backward_vector():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x.sum()
    y.backward()
    assert close(y.data, 6.0)
    assert close(x.grad, np.ones(3))


def test_mean_backward_vector():
    x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    y = x.mean()
    y.backward()
    assert close(y.data, 2.5)
    assert close(x.grad, np.ones(4) / 4)


def test_non_scalar_backward_requires_grad_arg():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x + 1
    with pytest.raises(ValueError):
        y.backward()


def test_manual_grad_for_non_scalar():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * 3
    y.backward(np.array([10.0, 20.0]))
    # dy/dx = 3; so grad = upstream * 3
    assert close(x.grad, np.array([30.0, 60.0]))


def test_broadcast_add_backward():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([10.0, 20.0], requires_grad=True)
    y = x + b
    s = y.sum()
    s.backward()
    assert close(x.grad, np.ones((2, 2)))
    # b broadcast over rows: each column appears twice
    assert close(b.grad, np.array([2.0, 2.0]))


def test_broadcast_mul_backward():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([10.0, 20.0], requires_grad=True)
    y = x * b
    s = y.sum()
    s.backward()
    # ds/dx = b broadcast to rows
    assert close(x.grad, np.array([[10.0, 20.0], [10.0, 20.0]]))
    # ds/db = sum over rows of x
    assert close(b.grad, np.array([1.0 + 3.0, 2.0 + 4.0]))


def test_matmul_forward_and_backward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    y = a.matmul(b)
    s = y.sum()
    s.backward()

    # forward
    assert close(y.data, np.array([[19.0, 22.0], [43.0, 50.0]]))

    # ds/da = ones @ b^T
    ones = np.ones_like(y.data)
    expected_da = ones @ b.data.T
    expected_db = a.data.T @ ones
    assert close(a.grad, expected_da)
    assert close(b.grad, expected_db)


def test_transpose_backward():
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)  # shape (1,3)
    y = x.T()  # (3,1)
    s = y.sum()
    s.backward()
    assert close(x.grad, np.ones_like(x.data))


def test_neg_and_sub():
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(5.0, requires_grad=True)
    z = y - x  # 3
    z.backward()
    assert close(z.data, 3.0)
    assert close(x.grad, -1.0)
    assert close(y.grad, 1.0)


def test_grad_accumulates_when_backward_called_twice():
    x = Tensor(2.0, requires_grad=True)
    y1 = x * 3
    y1.backward()
    assert close(x.grad, 3.0)

    y2 = x * 4
    y2.backward()
    assert close(x.grad, 7.0)  # accumulated: 3 + 4


def test_two_paths_to_same_leaf_accumulate_correctly():
    x = Tensor(2.0, requires_grad=True)
    y = x * x + x  # x^2 + x
    y.backward()
    # dy/dx = 2x + 1 = 5
    assert close(x.grad, 5.0)


def test_numeric_gradient_check_simple():
    # f(x) = sum((x * 3 + 1)^2)
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(2, 3))
    x = Tensor(x0, requires_grad=True)
    f = ((x * 3 + 1) ** 2).sum()
    f.backward()
    analytic = x.grad.copy()

    eps = 1e-6
    num = np.zeros_like(x0)
    for i in range(x0.shape[0]):
        for j in range(x0.shape[1]):
            xp = x0.copy()
            xm = x0.copy()
            xp[i, j] += eps
            xm[i, j] -= eps

            fp = (((xp * 3 + 1) ** 2).sum())
            fm = (((xm * 3 + 1) ** 2).sum())
            num[i, j] = (fp - fm) / (2 * eps)

    assert np.allclose(analytic, num, atol=1e-4, rtol=0)
