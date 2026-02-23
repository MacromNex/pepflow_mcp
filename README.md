# PEPFlow MCP

> MCP tools for optimization algorithm convergence analysis using PEPFlow framework

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Local Usage (Scripts)](#local-usage-scripts)
- [MCP Server Installation](#mcp-server-installation)
- [Using with Claude Code](#using-with-claude-code)
- [Using with Gemini CLI](#using-with-gemini-cli)
- [Available Tools](#available-tools)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

PEPFlow MCP provides tools for analyzing optimization algorithm convergence using the Performance Estimation Problem (PEP) framework. PEP is a mathematical methodology that enables worst-case analysis of first-order optimization methods for smooth convex functions.

### Features
- **Gradient Descent Analysis**: O(1/k) convergence verification with analytical proofs
- **Accelerated Gradient Methods**: O(1/k²) convergence with theta sequence computation
- **Algorithm Comparison**: Performance benchmarking across multiple optimization methods
- **Batch Processing**: Parameter sweeps and sensitivity analysis for algorithm tuning

### Directory Structure
```
./
├── README.md                           # This file
├── env/                                # Conda environment
├── src/
│   ├── server.py                       # MCP server (15 tools)
│   └── jobs/                          # Job management system
├── scripts/
│   ├── gradient_descent_analysis.py   # O(1/k) convergence analysis
│   ├── accelerated_gradient_analysis.py # O(1/k²) AGM analysis
│   ├── pep_optimization_framework.py  # Multi-algorithm framework
│   └── lib/                           # Shared utilities (45 functions)
├── examples/
│   ├── use_case_1_gradient_descent.py # Example GD analysis
│   ├── use_case_2_accelerated_gradient.py # Example AGM analysis
│   ├── use_case_3_pep_optimization.py # Example comparison analysis
│   └── data/                          # Algorithm notebooks and examples
├── configs/                           # Configuration files
│   ├── gradient_descent_config.json   # GD parameters
│   ├── accelerated_gradient_config.json # AGM parameters
│   └── pep_optimization_config.json   # General framework config
├── reports/                           # Step-by-step documentation
└── repo/                              # Original PEPFlow repository
    └── PEPFlow/                       # PEPFlow framework source
```

---

## Installation

### Quick Setup

Run the automated setup script:

```bash
./quick_setup.sh
```

This will create the environment and install all dependencies automatically.

### Manual Setup (Advanced)

For manual installation or customization, follow these steps.

#### Prerequisites
- Conda or Mamba (mamba recommended for faster installation)
- Python 3.10+
- PEPFlow framework dependencies (NumPy, SymPy, CVXPY, Matplotlib)

#### Create Environment
Please follow the environment setup procedure from `reports/step3_environment.md`. A verified workflow is:

```bash
# Navigate to the MCP directory
cd /home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/pepflow_mcp

# Create conda environment (use mamba if available)
mamba create -p ./env python=3.10 -y
# or: conda create -p ./env python=3.10 -y

# Activate environment
mamba activate ./env
# or: conda activate ./env

# Install PEPFlow Framework
cd repo/PEPFlow
mamba run -p ../../env pip install -e .

# Install MCP dependencies
cd ../../
mamba run -p ./env pip install fastmcp loguru --ignore-installed

# Install mathematical solvers
mamba install -c conda-forge osqp clarabel scs -y
```

---

## Local Usage (Scripts)

You can use the scripts directly without MCP for local processing.

### Available Scripts

| Script | Description | Algorithm | Convergence Rate |
|--------|-------------|-----------|------------------|
| `gradient_descent_analysis.py` | Gradient descent convergence analysis | Classic GD with stepsize 1/L | O(1/k) |
| `accelerated_gradient_analysis.py` | Nesterov's accelerated gradient method | AGM with theta sequences | O(1/k²) |
| `pep_optimization_framework.py` | General optimization framework | Multiple algorithms | Variable |

### Script Examples

#### Gradient Descent Analysis

```bash
# Activate environment
mamba activate ./env

# Run gradient descent analysis
mamba run -p ./env python scripts/gradient_descent_analysis.py \
  --config configs/gradient_descent_config.json \
  --iterations 8 \
  --proof-steps 2 \
  --output results/gd_analysis
```

**Parameters:**
- `--config, -c`: Configuration file path (default: uses built-in config)
- `--iterations, -n`: Number of iterations for numerical analysis (default: 8)
- `--proof-steps, -p`: Number of steps for analytical proof (default: 2)
- `--lipschitz, -L`: Lipschitz constant L (default: 1.0)
- `--radius, -R`: Initial distance bound R (default: 1.0)
- `--output, -o`: Output directory (default: results/)
- `--skip-plot`: Skip plotting for faster execution

#### Accelerated Gradient Analysis

```bash
mamba run -p ./env python scripts/accelerated_gradient_analysis.py \
  --config configs/accelerated_gradient_config.json \
  --iterations 5 \
  --proof-steps 3 \
  --output results/agm_analysis
```

**Special Features:**
- Theta sequence computation: θₖ = (1 + √(1 + 4θₖ₋₁²))/2
- O(1/k²) convergence verification
- Dual variable analysis

#### Algorithm Comparison

```bash
# Single algorithm
mamba run -p ./env python scripts/pep_optimization_framework.py \
  --algorithm gradient_descent \
  --iterations 5

# Heavy ball with momentum
mamba run -p ./env python scripts/pep_optimization_framework.py \
  --algorithm heavy_ball \
  --iterations 4 \
  --momentum 0.3

# Compare multiple algorithms
mamba run -p ./env python scripts/pep_optimization_framework.py \
  --algorithm comparison \
  --iterations 3
```

---

## MCP Server Installation

### Option 1: Using fastmcp (Recommended)

```bash
# Install MCP server for Claude Code
fastmcp install src/server.py --name pepflow-tools
```

### Option 2: Manual Installation for Claude Code

```bash
# Add MCP server to Claude Code
claude mcp add pepflow-tools -- $(pwd)/env/bin/python $(pwd)/src/server.py

# Verify installation
claude mcp list
```

### Option 3: Configure in settings.json

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "pepflow-tools": {
      "command": "/home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/pepflow_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/pepflow_mcp/src/server.py"]
    }
  }
}
```

---

## Using with Claude Code

After installing the MCP server, you can use it directly in Claude Code.

### Quick Start

```bash
# Start Claude Code
claude
```

### Example Prompts

#### Tool Discovery
```
What optimization analysis tools are available from pepflow-tools?
```

#### Quick Convergence Analysis (Sync)
```
Analyze gradient descent convergence for 8 iterations with 2 proof steps. Verify the O(1/k) convergence rate.
```

#### Large-Scale Analysis (Submit API)
```
Submit a gradient descent analysis job for 50 iterations and 20 proof steps. Name the job "large_scale_gd" and track its progress.
```

#### Check Job Status
```
Check the status of job abc12345 and show me the execution logs.
```

#### Algorithm Comparison
```
Compare the performance of gradient descent, heavy ball (momentum=0.3), and accelerated gradient methods for 10 iterations each.
```

#### Parameter Sweep
```
Submit a parameter sweep job for the heavy_ball algorithm with:
- momentum values: [0.1, 0.3, 0.5, 0.7, 0.9]
- lipschitz_constant values: [0.5, 1.0, 2.0]
Run 15 iterations for each combination.
```

### Using @ References

In Claude Code, use `@` to reference files and directories:

| Reference | Description |
|-----------|-------------|
| `@examples/use_case_1_gradient_descent.py` | Reference example GD analysis |
| `@configs/gradient_descent_config.json` | Reference a config file |
| `@results/` | Reference output directory |
| `@scripts/lib/pepflow_utils.py` | Reference utility functions |

---

## Using with Gemini CLI

### Configuration

Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "pepflow-tools": {
      "command": "/home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/pepflow_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/CycPepMCP/CycPepMCP/tool-mcps/pepflow_mcp/src/server.py"]
    }
  }
}
```

### Example Prompts

```bash
# Start Gemini CLI
gemini

# Example prompts (same as Claude Code)
> What optimization tools are available?
> Analyze gradient descent for 5 iterations
> Submit accelerated gradient analysis with 15 iterations
```

---

## Available Tools

### Quick Operations (Sync API)

These tools return results immediately (< 2 minutes):

| Tool | Description | Algorithm | Est. Runtime |
|------|-------------|-----------|--------------|
| `analyze_gradient_descent` | O(1/k) convergence analysis | Gradient Descent | 5-30 seconds |
| `analyze_accelerated_gradient` | O(1/k²) convergence with theta computation | Nesterov AGM | 8-60 seconds |
| `analyze_optimization_algorithm` | General algorithm framework | Multiple | 5-120 seconds |

### Long-Running Tasks (Submit API)

These tools return a job_id for tracking (> 2 minutes):

| Tool | Description | Default Params | Use Case |
|------|-------------|----------------|----------|
| `submit_gradient_descent_analysis` | Large-scale GD analysis | 20 iter, 10 proofs | Extensive verification |
| `submit_accelerated_gradient_analysis` | Extensive AGM analysis | 15 iter, 8 proofs | Complex theta computation |
| `submit_optimization_comparison` | Multi-algorithm comparison | 10 iter, 3 algorithms | Performance benchmarking |
| `submit_parameter_sweep` | Parameter optimization | Variable ranges | Sensitivity analysis |

### Job Management Tools

| Tool | Description | Usage |
|------|-------------|-------|
| `get_job_status` | Check job progress | Monitor running jobs |
| `get_job_result` | Get completed results | Retrieve analysis when done |
| `get_job_log` | View execution logs | Debug failed jobs |
| `cancel_job` | Cancel running job | Stop unwanted jobs |
| `list_jobs` | List all jobs | Overview of job queue |

### Utility Tools

| Tool | Description | Returns |
|------|-------------|---------|
| `get_available_algorithms` | List supported algorithms | Algorithm info with convergence rates |
| `get_default_configs` | Get default parameters | Configuration templates |
| `validate_pepflow_installation` | Check framework | Installation status and versions |

---

## Examples

### Example 1: Quick Convergence Verification

**Goal:** Verify O(1/k) convergence rate for gradient descent

**Using Script:**
```bash
mamba run -p ./env python scripts/gradient_descent_analysis.py \
  --iterations 8 \
  --proof-steps 2 \
  --skip-plot \
  --output results/quick_verification
```

**Using MCP (in Claude Code):**
```
Analyze gradient descent convergence for 8 iterations with 2 proof steps. Show me if the O(1/k) theoretical bound is verified.
```

**Expected Output:**
- Numerical convergence values: [0.166670, 0.099995, ...]
- Analytical bound: L/(4N+2) ≈ 0.1000000000
- Convergence verified: true/false
- Optimal function value progression

### Example 2: Accelerated Gradient Method Analysis

**Goal:** Analyze O(1/k²) convergence with theta sequence computation

**Using Script:**
```bash
mamba run -p ./env python scripts/accelerated_gradient_analysis.py \
  --iterations 5 \
  --proof-steps 3 \
  --output results/agm_analysis
```

**Using MCP (in Claude Code):**
```
Submit accelerated gradient analysis for 15 iterations and 8 proof steps. Track the job progress and show me the theta sequence when complete.
```

**Expected Output:**
- Theta sequence: [1.618034, 2.193527, 2.701588, ...]
- O(1/k²) bound: L/(2*θₙ²) verification
- Convergence plots and analytical verification

### Example 3: Algorithm Performance Comparison

**Goal:** Compare multiple optimization methods for benchmarking

**Using MCP (in Claude Code):**
```
Compare the performance of these optimization algorithms for 10 iterations each:
1. Gradient descent
2. Heavy ball with momentum 0.5
3. Accelerated gradient method

Show me which achieves the best convergence rate and at which iteration.
```

**Expected Output:**
- Comparative convergence values
- Best algorithm identification
- Performance metrics per iteration
- Convergence rate verification for each method

---

## Demo Data

The `examples/data/` directory contains algorithm notebooks and examples:

| Directory | Description | Use With |
|-----------|-------------|----------|
| `gd/` | Gradient descent examples | `analyze_gradient_descent` |
| `agm/` | Accelerated gradient examples | `analyze_accelerated_gradient` |
| `pgm/` | Proximal gradient examples | Future extensions |
| `drs/` | Douglas-Rachford splitting | Future extensions |
| `ogm/` | Optimized gradient methods | Future extensions |

---

## Configuration Files

The `configs/` directory contains configuration templates:

| Config | Algorithm | Key Parameters |
|--------|-----------|----------------|
| `gradient_descent_config.json` | Gradient Descent | iterations, proof_steps, L, R |
| `accelerated_gradient_config.json` | Nesterov AGM | theta computation, dual plots |
| `pep_optimization_config.json` | General Framework | algorithm selection, comparison mode |

### Config Example

```json
{
  "_description": "Gradient descent convergence analysis configuration",
  "_source": "examples/use_case_1_gradient_descent.py",

  "algorithm": {
    "name": "gradient_descent",
    "stepsize": "1/L"
  },
  "analysis": {
    "iterations": 8,
    "proof_steps": 2
  },
  "parameters": {
    "lipschitz_constant": 1.0,
    "initial_radius": 1.0
  },
  "output": {
    "save_plot": true,
    "output_dir": "results/gradient_descent"
  },
  "numerical_settings": {
    "solver": "OSQP",
    "precision": 1e-10
  }
}
```

---

## Troubleshooting

### Environment Issues

**Problem:** Environment not found
```bash
# Recreate environment
mamba create -p ./env python=3.10 -y
mamba activate ./env
cd repo/PEPFlow && mamba run -p ../../env pip install -e .
cd ../../ && mamba run -p ./env pip install fastmcp loguru
```

**Problem:** PEPFlow import errors
```bash
# Verify PEPFlow installation
mamba run -p ./env python -c "import pepflow; print('PEPFlow: OK')"

# If it fails, reinstall
cd repo/PEPFlow
mamba run -p ../../env pip install -e . --force-reinstall
```

**Problem:** Solver errors (OSQP, SCS, Clarabel)
```bash
# Install mathematical solvers
mamba activate ./env
mamba install -c conda-forge osqp clarabel scs -y
```

### MCP Issues

**Problem:** Server not found in Claude Code
```bash
# Check MCP registration
claude mcp list

# Re-add if needed
claude mcp remove pepflow-tools
claude mcp add pepflow-tools -- $(pwd)/env/bin/python $(pwd)/src/server.py
```

**Problem:** Tools not working
```bash
# Test server directly
mamba run -p ./env python -c "
from src.server import mcp
print(list(mcp.list_tools().keys()))
print('Available tools:', len(list(mcp.list_tools().keys())))
"
```

**Problem:** Import errors in server
```bash
# Verify all dependencies
mamba run -p ./env python -c "
import pepflow, sympy, numpy, matplotlib
print('All dependencies available')
"
```

### Job Issues

**Problem:** Job stuck in pending
```bash
# Check job directory
ls -la jobs/

# Check if job manager is working
mamba run -p ./env python -c "
from src.jobs.manager import job_manager
print('Jobs directory:', job_manager.jobs_dir)
"
```

**Problem:** Job failed
```
Use get_job_log with job_id "abc12345" and tail 100 to see error details in the execution logs
```

**Problem:** Analysis fails with convergence errors
- Try reducing iterations or proof_steps
- Check if Lipschitz constant is appropriate (try L=1.0)
- Verify solver is working: `mamba run -p ./env python -c "import cvxpy; print(cvxpy.installed_solvers())"`

### Algorithm-Specific Issues

**Problem:** Theta sequence computation fails (AGM)
- Reduce proof_steps (try 2-5)
- Check numerical precision settings
- Verify SymPy is working for symbolic computation

**Problem:** Dual variable extraction errors
- This is common for high iteration counts
- Scripts include error handling for graceful degradation
- Results are still valid even if dual variables can't be extracted

---

## Development

### Running Tests

```bash
# Activate environment
mamba activate ./env

# Test basic functionality
mamba run -p ./env python test_simple.py

# Test MCP server startup
mamba run -p ./env python test_server_startup.py

# Test individual scripts
mamba run -p ./env python scripts/gradient_descent_analysis.py --help
```

### Starting Dev Server

```bash
# Run MCP server in development mode
mamba run -p ./env fastmcp dev src/server.py

# Or production mode
mamba run -p ./env python src/server.py
```

---

## Mathematical Background

### PEP Framework
Performance Estimation Problems (PEP) provide a systematic way to analyze worst-case convergence of optimization algorithms by formulating the analysis as a semidefinite program.

### Supported Algorithms

| Algorithm | Mathematical Form | Convergence Rate | Parameters |
|-----------|-------------------|------------------|------------|
| Gradient Descent | xₖ₊₁ = xₖ - (1/L)∇f(xₖ) | O(1/k) | stepsize = 1/L |
| Heavy Ball | xₖ₊₁ = xₖ - (1/L)∇f(xₖ) + β(xₖ - xₖ₋₁) | O(1/k) | momentum β |
| Accelerated Gradient | AGM with optimal theta sequence | O(1/k²) | adaptive momentum |

### Convergence Bounds

- **Gradient Descent**: f(xₖ) - f* ≤ L/(4k+2) · R²
- **Accelerated Gradient**: f(xₖ) - f* ≤ L/(2θₖ²) · R²

where L is the Lipschitz constant and R is the initial distance bound.

---

## License

MIT License - Based on the PEPFlow framework

## Credits

Based on [PEPFlow](https://github.com/cvxgrp/pepflow) by CVX Research Group