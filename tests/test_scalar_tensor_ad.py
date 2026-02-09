import sys
import unittest
import math
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "repository_after"))
from scalar_tensor_ad import Tensor

class TestTensor(unittest.TestCase):
    
    def test_basic_arithmetic(self):
        """Test basic op correctness and simple backprop."""
        # Add
        x, y = Tensor(3.0), Tensor(2.0)
        self.assertEqual((x + y).data, 5.0)
        (x + y).backward(); self.assertEqual(x.grad, 1.0)
        
        # Sub
        x, y = Tensor(5.0), Tensor(3.0)
        self.assertEqual((x - y).data, 2.0)
        (x - y).backward(); self.assertEqual(y.grad, -1.0)
        
        # Mul
        x, y = Tensor(4.0), Tensor(3.0)
        self.assertEqual((x * y).data, 12.0)
        (x * y).backward(); self.assertEqual(x.grad, 3.0)
        
        # Div
        x, y = Tensor(10.0), Tensor(2.0)
        self.assertEqual((x / y).data, 5.0)
        (x / y).backward(); self.assertEqual(y.grad, -2.5)

    def test_r_ops_and_scalars(self):
        """Test magic methods for scalar interactions (radd, rsub, etc)."""
        x = Tensor(2.0)
        self.assertEqual((3.0 + x).data, 5.0)
        self.assertEqual((10.0 - x).data, 8.0)
        self.assertEqual((3.0 * x).data, 6.0)
        self.assertEqual((1.0 / x).data, 0.5)
        
        (1.0 / x).backward()
        self.assertEqual(x.grad, -0.25)

    def test_power(self):
        """Test power operator and gradients."""
        x = Tensor(3.0)
        (x ** 3).backward()
        self.assertEqual(x.grad, 27.0)
        
        y = Tensor(4.0)
        (y ** 0.5).backward()
        self.assertEqual(y.grad, 0.25)

    def test_activations(self):
        """Test tanh, relu, and sigmoid."""
        x = Tensor(0.0)
        self.assertEqual(x.tanh().data, 0.0)
        self.assertEqual(x.relu().data, 0.0)
        self.assertEqual(x.sigmoid().data, 0.5)
        
        # Grads at 0
        x = Tensor(0.0); x.tanh().backward();    self.assertEqual(x.grad, 1.0)
        x = Tensor(0.0); x.relu().backward();    self.assertEqual(x.grad, 0.0)
        x = Tensor(0.0); x.sigmoid().backward(); self.assertEqual(x.grad, 0.25)

    def test_gradient_accumulation(self):
        """Verify nodes in multiple paths sum gradients."""
        x = Tensor(2.0)
        # y = x^2 + x^3 => dy/dx = 2x + 3x^2 = 4 + 12 = 16
        y = x**2 + x**3
        y.backward()
        self.assertEqual(x.grad, 16.0)

    def test_complex_graph(self):
        """Full system test with complex expression."""
        x = Tensor(-4.0)
        z = 2 * x + 2 + x
        q = z.relu() + z.sigmoid()
        # z = -4*3 + 2 = -10
        # q = relu(-10) + sig(-10) = 0 + sig(-10)
        self.assertAlmostEqual(q.data, 1 / (1 + math.exp(10)))
        
        q.backward()
        # dq/dz = (relu' at -10) + (sig' at -10) = 0 + sig(-10)*(1-sig(-10))
        # dz/dx = 3
        # dq/dx = 3 * ds_dz
        s = 1 / (1 + math.exp(10))
        expected_grad = 3 * (s * (1 - s))
        self.assertAlmostEqual(x.grad, expected_grad)

    def test_numerical_stability(self):
        """Test extremes and long chains."""
        # Deep chain: x * 2 * 2 * ...
        x = Tensor(1.0)
        y = x
        for _ in range(5): y = y * 2.0
        y.backward()
        self.assertEqual(x.grad, 32.0)
        
        # Stability with small numbers
        x = Tensor(1e-4)
        y = x.sigmoid()
        y.backward()
        self.assertTrue(x.grad > 0)

    def test_nested_activations(self):
        """Test chains of activations like relu(tanh(x))."""
        x = Tensor(0.5)
        y = x.tanh().relu()
        y.backward()
        # dy/dx = relu'(tanh(x)) * tanh'(x)
        # since tanh(0.5) > 0, relu'(...) = 1
        # dy/dx = 1 * (1 - tanh(0.5)^2)
        expected = 1 - math.tanh(0.5)**2
        self.assertAlmostEqual(x.grad, expected)

    def test_misc_coverage(self):
        """Ensure __repr__ and other minor methods are covered."""
        t = Tensor(1.0)
        self.assertIn("data=1.0000", repr(t))
        # Topological order stability check (diamond)
        a = Tensor(2.0)
        b, c = a*2, a*3
        (b + c).backward()
        self.assertEqual(a.grad, 5.0)

    def test_edge_cases_and_nesting(self):
        """Ultra-detailed edge cases for every operation."""
        # x - x grad check
        x = Tensor(5.0)
        z = x - x
        z.backward()
        self.assertEqual(x.grad, 0.0) # 1 - 1 = 0
        
        # x / x grad check
        x = Tensor(5.0)
        z = x / x
        z.backward()
        # d(x/x)/dx = (1 * x - x * 1) / x^2 = 0
        self.assertEqual(x.grad, 0.0)
        
        # Power of Power
        x = Tensor(2.0)
        z = (x ** 2) ** 3 # x^6
        z.backward()
        # dz/dx = 6 * x^5 = 6 * 32 = 192
        self.assertEqual(x.grad, 192.0)
        
        # Nested Activations with Arithmetic
        x = Tensor(0.5)
        z = (x.sigmoid() * 2.0).relu()
        z.backward()
        # s = sig(0.5). z = 2*s if 2*s>0 else 0.
        # dz/dx = relu'(2*s) * 2 * sig'(0.5) = 1 * 2 * s * (1-s)
        s = 1 / (1 + math.exp(-0.5))
        self.assertAlmostEqual(x.grad, 2 * s * (1 - s))

if __name__ == '__main__':
    unittest.main()
