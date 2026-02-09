import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'repository_after'))
from tensor import Tensor
import math


def test_square_gradient():
    x = Tensor(3.0)
    y = x * x
    y.backward()
    assert abs(x.grad - 6.0) < 1e-6


def test_chain_rule():
    x = Tensor(2.0)
    y = (x * x + x).tanh()
    y.backward()

   
    t = math.tanh(2 * 2 + 2)
    expected = (1 - t ** 2) * (2 * 2 + 1)

    assert abs(x.grad - expected) < 1e-6


def test_gradient_accumulation():
    x = Tensor(5.0)
    y = x + x + x
    y.backward()
    assert x.grad == 3.0


def test_relu():
    x = Tensor(-3.0)
    y = x.relu()
    y.backward()
    assert x.grad == 0.0

    x2 = Tensor(3.0)
    y2 = x2.relu()
    y2.backward()
    assert x2.grad == 1.0


def test_sigmoid():
    x = Tensor(0.0)
    y = x.sigmoid()
    y.backward()

    expected = 0.25  
    assert abs(x.grad - expected) < 1e-6


def run_all_tests():
    """Run all tests and report results"""
    print("=" * 60)
    print("Running Automatic Differentiation Engine Tests")
    print("=" * 60)
    print()
    
    tests = [
        ("test_square_gradient", test_square_gradient),
        ("test_chain_rule", test_chain_rule),
        ("test_gradient_accumulation", test_gradient_accumulation),
        ("test_relu", test_relu),
        ("test_sigmoid", test_sigmoid),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"[PASS] {test_name}")
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {test_name}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {test_name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Tests passed: {passed}/{len(tests)}")
    if failed == 0:
        print("All tests passed! [SUCCESS]")
    else:
        print(f"Tests failed: {failed}")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
