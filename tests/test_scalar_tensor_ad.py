import unittest
import math
import sys

from scalar_tensor_ad import Tensor

class TestTensorDetailed(unittest.TestCase):
    
    def test_arithmetic_fwd_bwd(self):
        # Addition
        a, b = Tensor(1.5), Tensor(2.5)
        out = a + b
        self.assertEqual(out.data, 4.0)
        out.backward()
        self.assertEqual(a.grad, 1.0)
        self.assertEqual(b.grad, 1.0)

        # Subtraction
        a, b = Tensor(5.0), Tensor(2.0)
        out = a - b
        self.assertEqual(out.data, 3.0)
        out.backward()
        self.assertEqual(a.grad, 1.0)
        self.assertEqual(b.grad, -1.0)

        # Multiplication
        a, b = Tensor(3.0), Tensor(4.0)
        out = a * b
        self.assertEqual(out.data, 12.0)
        out.backward()
        self.assertEqual(a.grad, 4.0)
        self.assertEqual(b.grad, 3.0)

        # Division
        a, b = Tensor(10.0), Tensor(2.0)
        out = a / b
        self.assertEqual(out.data, 5.0)
        out.backward()
        self.assertEqual(a.grad, 0.5)
        self.assertEqual(b.grad, -2.5)

    def test_scalar_interactions(self):
        x = Tensor(4.0)
        self.assertEqual((x + 1).data, 5.0)
        self.assertEqual((1 + x).data, 5.0)
        self.assertEqual((x - 1).data, 3.0)
        self.assertEqual((5 - x).data, 1.0)
        self.assertEqual((x * 2).data, 8.0)
        self.assertEqual((2 * x).data, 8.0)
        self.assertEqual((x / 2).data, 2.0)
        self.assertEqual((8 / x).data, 2.0)

        (8 / x).backward()
        self.assertEqual(x.grad, -0.5)

    def test_power_variants(self):
        # Integer power
        x = Tensor(2.0)
        (x ** 3).backward()
        self.assertEqual(x.grad, 12.0)

        # Fractional power
        x = Tensor(9.0)
        (x ** 0.5).backward()
        self.assertEqual(x.grad, 1/6)

        # Negative power
        x = Tensor(2.0)
        (x ** -1).backward()
        self.assertEqual(x.grad, -0.25)

    def test_activations_detailed(self):
        # ReLU boundaries
        for v in [-1.0, 0.0, 1.0]:
            x = Tensor(v)
            out = x.relu()
            self.assertEqual(out.data, max(0.0, v))
            out.backward()
            self.assertEqual(x.grad, 1.0 if v > 0.0 else 0.0)

        # Tanh check
        x = Tensor(0.5)
        x.tanh().backward()
        self.assertAlmostEqual(x.grad, 1 - math.tanh(0.5)**2)

        # Sigmoid check
        x = Tensor(-0.5)
        s = 1 / (1 + math.exp(0.5))
        x.sigmoid().backward()
        self.assertAlmostEqual(x.grad, s * (1 - s))

    def test_multi_path_accumulation(self):
        # f(x) = x + x + x * x
        x = Tensor(3.0)
        z = x + x + x * x
        # dz/dx = 1 + 1 + 2x = 2 + 6 = 8
        z.backward()
        self.assertEqual(x.grad, 8.0)

        # f(x) = (x + 1) * (x + 2)
        x = Tensor(2.0)
        z = (x + 1) * (x + 2)
        # dz/dx = (x+2) + (x+1) = 2x + 3 = 7
        z.backward()
        self.assertEqual(x.grad, 7.0)

    def test_shared_subexpression(self):
        # a = x*y; b = a + x; c = a + y; out = b*c
        x, y = Tensor(2.0), Tensor(3.0)
        a = x * y    # 6
        b = a + x    # 8
        c = a + y    # 9
        out = b * c  # 72
        out.backward()
        
        # da/dx = y=3, da/dy = x=2
        # db/da = 1, db/dx = 1
        # dc/da = 1, dc/dy = 1
        # dout/db = c=9, dout/dc = b=8
        
        # dout/da = (dout/db * db/da) + (dout/dc * dc/da) = 9*1 + 8*1 = 17
        # dout/dx = (dout/db * db/dx) + (dout/da * da/dx) = 9*1 + 17*3 = 9 + 51 = 60
        # dout/dy = (dout/dc * dc/dy) + (dout/da * da/dy) = 8*1 + 17*2 = 8 + 34 = 42
        
        self.assertEqual(x.grad, 60.0)
        self.assertEqual(y.grad, 42.0)

    def test_numerical_stability_extreme(self):
        # Small gradient flow
        x = Tensor(10.0)
        y = x.sigmoid() # grad ≈ 0
        y.backward()
        self.assertTrue(x.grad < 1e-4)

        # Chain multiplication
        x = Tensor(1.0)
        cur = x
        for _ in range(10):
            cur = cur * 2.0 # 2^10 = 1024
        cur.backward()
        self.assertEqual(x.grad, 1024.0)

    def test_graph_attributes(self):
        x = Tensor(1.0)
        y = Tensor(2.0)
        z = x + y
        self.assertEqual(z.op, "+")
        self.assertIn(x, z._prev)
        self.assertIn(y, z._prev)
        self.assertTrue("Tensor" in repr(z))

    def test_reverse_ops_detailed(self):
        # 5.0 - x
        x = Tensor(2.0)
        z = 5.0 - x
        self.assertEqual(z.data, 3.0)
        z.backward()
        self.assertEqual(x.grad, -1.0)
        
        # 10.0 / x
        x = Tensor(2.0)
        z = 10.0 / x
        self.assertEqual(z.data, 5.0)
        z.backward()
        # dz/dx = -10/x^2 = -10/4 = -2.5
        self.assertEqual(x.grad, -2.5)

    def test_middle_node_backward(self):
        # x -> a -> out
        x = Tensor(2.0)
        a = x * 3.0
        out = a.relu()
        # call backward on 'a' instead of 'out'
        a.backward()
        self.assertEqual(x.grad, 3.0)
        # 'out' grad should still be 0.0 as it's downstream
        self.assertEqual(out.grad, 0.0)

    def test_gradient_reset_accumulation(self):
        # Check that we can manually zero grad and re-run
        x = Tensor(2.0)
        y = x * x
        y.backward()
        self.assertEqual(x.grad, 4.0)
        
        # Another backward adds to it (standard behavior for this engine)
        y.backward()
        self.assertEqual(x.grad, 8.0)
        
        # Manual reset
        x.grad = 0.0
        y.backward()
        self.assertEqual(x.grad, 4.0)

    def test_complex_nesting(self):
        # f(x, y) = tanh( x*y + sigmoid(x) )
        x, y = Tensor(0.5), Tensor(1.0)
        z = (x * y + x.sigmoid()).tanh()
        z.backward()
        
        # Manual check
        # s = sig(0.5) = 0.622459
        # arg = 0.5 * 1.0 + s = 1.122459
        # val = tanh(1.122459) = 0.8084
        # dz/darg = 1 - tanh(arg)^2 = 1 - 0.6535 = 0.3465
        # darg/dx = y + sig'(x) = 1.0 + s*(1-s) = 1.0 + 0.235 = 1.235
        # dz/dx = 0.3465 * 1.235 = 0.4279
        self.assertAlmostEqual(z.data, math.tanh(0.5 + 1/(1+math.exp(-0.5))), places=5)
        self.assertAlmostEqual(x.grad, (1 - math.tanh(0.5 + 1/(1+math.exp(-0.5)))**2) * (1.0 + (1/(1+math.exp(-0.5))) * (1 - 1/(1+math.exp(-0.5)))), places=5)

if __name__ == "__main__":
    unittest.main()
