#!/bin/bash
set -e

cd "$(dirname "$0")/.."
cd repository_after
python -m pytest ../tests/ -v --tb=short > ../evaluation/results.txt 2>&1
