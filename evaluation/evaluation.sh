#!/bin/bash
# Evaluation script for Scalar Tensor AD Engine

echo "Starting evaluation..."

# 1. Run core logic
python repository_after/scalar_tensor_ad.py > output.log 2>&1
if [ $? -ne 0 ]; then
    echo "Execution failed!"
    exit 1
fi

# 2. Run unit tests
python -m unittest discover tests > test_results.log 2>&1
if [ $? -ne 0 ]; then
    echo "Tests failed!"
    exit 1
fi

echo "All checks passed!"
exit 0
