#!/bin/bash

echo "====================================="
echo " Running Automatic Differentiation Evaluation "
echo "====================================="

# Build the Docker image
docker-compose build

# Run the tests inside the container
docker-compose run --rm autodiff python -m tests.test_tensor

echo "====================================="
echo " Evaluation completed successfully "
echo "====================================="
