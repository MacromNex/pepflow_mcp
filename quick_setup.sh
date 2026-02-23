#!/bin/bash
# Quick Setup Script for PEPFlow MCP
# PEPFlow: Performance Estimation Problem Workflow
# Framework for analyzing convergence of optimization algorithms
# Source: https://github.com/pepflow-lib/PEPFlow

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up PEPFlow MCP ==="

# Step 1: Create Python environment
echo "[1/5] Creating Python 3.10 environment..."
(command -v mamba >/dev/null 2>&1 && mamba create -p ./env python=3.10 -y) || \
(command -v conda >/dev/null 2>&1 && conda create -p ./env python=3.10 -y) || \
(echo "Warning: Neither mamba nor conda found, creating venv instead" && python3 -m venv ./env)

# Step 2: Install PEPFlow from repo
echo "[2/5] Installing PEPFlow..."
cd repo/PEPFlow && ../../env/bin/pip install -e . && cd ../..

# Step 3: Install loguru
echo "[3/5] Installing loguru..."
./env/bin/pip install loguru --ignore-installed

# Step 4: Install fastmcp
echo "[4/5] Installing fastmcp..."
./env/bin/pip install --ignore-installed fastmcp

# Step 5: Install optimization solvers
echo "[5/5] Installing optimization solvers..."
(command -v mamba >/dev/null 2>&1 && mamba install -p ./env -c conda-forge osqp clarabel scs -y) || \
(command -v conda >/dev/null 2>&1 && conda install -p ./env -c conda-forge osqp clarabel scs -y) || \
(echo "Warning: Neither mamba nor conda found for solver installation, installing via pip" && ./env/bin/pip install osqp clarabel scs)

echo ""
echo "=== PEPFlow MCP Setup Complete ==="
echo "For documentation, visit: https://pepflow-lib.github.io/PEPFlow/"
echo "To run the MCP server: ./env/bin/python src/server.py"
