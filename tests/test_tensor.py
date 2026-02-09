import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'repository_after'))

from tensor import Tensor

def test_basic_operations():
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = a + b
    assert c.data == 5.0
    
    d = a * b
    assert d.data == 6.0
    
    e = a - b
    assert abs(e.data - (-1.0)) < 1e-6
    
    f = a / b
    assert abs(f.data - (2.0/3.0)) < 1e-6

def test_addition_gradient():
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = a + b
    c.backward()
    
    assert abs(a.grad - 1.0) < 1e-6
    assert abs(b.grad - 1.0) < 1e-6

def test_multiplication_gradient():
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = a * b
    c.backward()
    
    assert abs(a.grad - 3.0) < 1e-6
    assert abs(b.grad - 2.0) < 1e-6

def test_power_gradient():
    a = Tensor(2.0)
    b = a ** 3
    b.backward()
    
    assert abs(b.data - 8.0) < 1e-6
    assert abs(a.grad - 12.0) < 1e-6

def test_division_gradient():
    a = Tensor(6.0)
    b = Tensor(2.0)
    c = a / b
    c.backward()
    
    assert abs(c.data - 3.0) < 1e-6
    assert abs(a.grad - 0.5) < 1e-6
    assert abs(b.grad - (-1.5)) < 1e-6

def test_tanh():
    a = Tensor(0.0)
    b = a.tanh()
    b.backward()
    
    assert abs(b.data - 0.0) < 1e-6
    assert abs(a.grad - 1.0) < 1e-6

def test_relu():
    a = Tensor(-1.0)
    b = a.relu()
    b.backward()
    
    assert b.data == 0.0
    assert a.grad == 0.0
    
    c = Tensor(2.0)
    d = c.relu()
    d.backward()
    
    assert d.data == 2.0
    assert c.grad == 1.0

def test_sigmoid():
    a = Tensor(0.0)
    b = a.sigmoid()
    b.backward()
    
    assert abs(b.data - 0.5) < 1e-6
    assert abs(a.grad - 0.25) < 1e-6

def test_complex_expression():
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = a * b + a
    c.backward()
    
    assert abs(c.data - 8.0) < 1e-6
    assert abs(a.grad - 4.0) < 1e-6
    assert abs(b.grad - 2.0) < 1e-6

def test_gradient_accumulation():
    a = Tensor(2.0)
    b = a + a
    b.backward()
    
    assert abs(b.data - 4.0) < 1e-6
    assert abs(a.grad - 2.0) < 1e-6

def test_chain_rule():
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = a * b
    d = c * a
    d.backward()
    
    assert abs(d.data - 12.0) < 1e-6
    assert abs(a.grad - 12.0) < 1e-6
    assert abs(b.grad - 4.0) < 1e-6

def test_scalar_operations():
    a = Tensor(2.0)
    b = a + 3.0
    c = 5.0 * a
    b.backward()
    
    assert abs(b.data - 5.0) < 1e-6
    assert abs(a.grad - 1.0) < 1e-6
    
    a.grad = 0.0
    c.backward()
    assert abs(c.data - 10.0) < 1e-6
    assert abs(a.grad - 5.0) < 1e-6

if __name__ == '__main__':
    test_basic_operations()
    test_addition_gradient()
    test_multiplication_gradient()
    test_power_gradient()
    test_division_gradient()
    test_tanh()
    test_relu()
    test_sigmoid()
    test_complex_expression()
    test_gradient_accumulation()
    test_chain_rule()
    test_scalar_operations()
    print("All tests passed!")

