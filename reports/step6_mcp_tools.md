# Step 6: MCP Tools Documentation

## Server Information
- **Server Name**: optimization-analysis
- **Version**: 1.0.0
- **Created Date**: 2025-12-31
- **Server Path**: `src/server.py`
- **Framework**: FastMCP 2.14.2
- **Dependencies**: PEPFlow framework, NumPy, SymPy, Matplotlib

## API Design Analysis

Based on the analysis of Step 5 scripts, the following API classifications were determined:

### Script Runtime Analysis

| Script | Est. Runtime | API Type | Reasoning |
|--------|-------------|----------|-----------|
| `gradient_descent_analysis.py` | ~5-30 sec | **Sync** (with Submit option) | Fast for small iterations; Submit for large-scale |
| `accelerated_gradient_analysis.py` | ~8-60 sec | **Sync** (with Submit option) | Theta computation can be intensive |
| `pep_optimization_framework.py` | ~5-120 sec | **Sync** (with Submit option) | Comparison mode can be slow |

### API Strategy
- **Sync API**: For iterations ≤ 10, proof_steps ≤ 5 (typical completion < 2 minutes)
- **Submit API**: For iterations > 10, extensive analysis, or batch processing
- **Both available**: Users can choose based on their needs

---

## Job Management Tools

| Tool | Description | Parameters | Returns |
|------|-------------|------------|---------|
| `get_job_status` | Check job progress | `job_id: str` | Status, timestamps, errors |
| `get_job_result` | Get completed job results | `job_id: str` | Analysis results or error |
| `get_job_log` | View job execution logs | `job_id: str, tail: int = 50` | Log lines and count |
| `cancel_job` | Cancel running job | `job_id: str` | Success/error message |
| `list_jobs` | List all jobs | `status: str = None` | Filtered job list |

### Job States
- **pending**: Job submitted, waiting to start
- **running**: Job currently executing
- **completed**: Job finished successfully
- **failed**: Job encountered an error
- **cancelled**: Job was cancelled by user

---

## Synchronous Tools (Fast Operations)

### 1. analyze_gradient_descent

**Description**: Analyze gradient descent convergence using PEPFlow framework
**Algorithm**: Gradient Descent with stepsize 1/L
**Convergence Rate**: O(1/k)
**Est. Runtime**: 5-30 seconds

**Parameters:**
```python
iterations: int = 8          # Number of iterations for numerical analysis
proof_steps: int = 2         # Number of steps for analytical proof
lipschitz_constant: float = 1.0  # Lipschitz constant L
initial_radius: float = 1.0  # Initial distance bound R
output_dir: str = None       # Optional output directory
skip_plot: bool = True       # Skip plotting for faster execution
config_file: str = None      # Optional config file path
```

**Returns:**
```python
{
  "status": "success",
  "numerical_results": [0.166670, 0.099995, ...],  # Convergence values
  "analytical_results": {...},                     # Proof verification
  "convergence_verified": true,                    # Whether bound satisfied
  "output_dir": "results/...",                     # Output directory path
  "metadata": {"algorithm": "gradient_descent", ...}
}
```

### 2. analyze_accelerated_gradient

**Description**: Analyze Nesterov's accelerated gradient method
**Algorithm**: Nesterov AGM with optimal theta sequence
**Convergence Rate**: O(1/k²)
**Est. Runtime**: 8-60 seconds

**Parameters:** (Similar to gradient descent, plus theta computation)

**Special Features:**
- Theta sequence computation: θ_k = (1 + √(1 + 4θ_{k-1}²))/2
- Cached computation for efficiency
- O(1/k²) convergence verification

### 3. analyze_optimization_algorithm

**Description**: General framework for analyzing multiple optimization algorithms
**Algorithms**: gradient_descent, heavy_ball, accelerated_gradient, comparison
**Est. Runtime**: 5-120 seconds (comparison mode can be slow)

**Parameters:**
```python
algorithm: str = "gradient_descent"  # Algorithm to analyze
iterations: int = 5                  # Number of iterations
momentum: float = 0.5                # Momentum for heavy_ball
lipschitz_constant: float = 1.0      # Lipschitz constant L
initial_radius: float = 1.0          # Initial radius R
output_dir: str = None               # Output directory
config_file: str = None              # Config file path
```

**Supported Algorithms:**
- `gradient_descent`: Classic GD with fixed stepsize
- `heavy_ball`: GD with momentum term
- `accelerated_gradient`: Simplified AGM
- `comparison`: Run multiple algorithms and compare

---

## Submit Tools (Long Operations)

### 1. submit_gradient_descent_analysis

**Description**: Submit gradient descent analysis for background processing
**Use Case**: Large iteration counts, extensive proof verification

**Parameters:**
```python
iterations: int = 20         # Higher default for background jobs
proof_steps: int = 10        # More proof steps
lipschitz_constant: float = 1.0
initial_radius: float = 1.0
output_dir: str = None
job_name: str = None         # Optional job name for tracking
config_file: str = None
```

**Returns:**
```python
{
  "status": "submitted",
  "job_id": "abc12345",
  "message": "Job submitted. Use get_job_status('abc12345') to check progress."
}
```

### 2. submit_accelerated_gradient_analysis

**Description**: Submit AGM analysis with extensive theta computation
**Use Case**: High iteration counts, complex analytical verification

**Parameters:** (Similar to sync version with higher defaults)

### 3. submit_optimization_comparison

**Description**: Submit comprehensive algorithm comparison job
**Use Case**: Multi-algorithm analysis, performance benchmarking

**Parameters:**
```python
algorithms: List[str] = ["gradient_descent", "heavy_ball", "accelerated_gradient"]
iterations: int = 10
momentum: float = 0.5
lipschitz_constant: float = 1.0
initial_radius: float = 1.0
output_dir: str = None
job_name: str = None
config_file: str = None
```

### 4. submit_parameter_sweep

**Description**: Submit parameter sweep analysis job
**Use Case**: Parameter optimization, sensitivity analysis

**Parameters:**
```python
algorithm: str = "gradient_descent"
parameter_ranges: Dict[str, List[float]] = None  # Parameter value ranges
base_iterations: int = 10
output_dir: str = None
job_name: str = None
```

**Example Parameter Ranges:**
```python
# For heavy_ball algorithm
{
  "lipschitz_constant": [0.5, 1.0, 2.0],
  "momentum": [0.1, 0.3, 0.5, 0.7, 0.9]
}

# For gradient_descent
{
  "lipschitz_constant": [0.5, 1.0, 1.5, 2.0],
  "initial_radius": [0.5, 1.0, 1.5]
}
```

---

## Utility Tools

### 1. get_available_algorithms

**Description**: Get list of available algorithms and their properties

**Returns:**
```python
{
  "status": "success",
  "algorithms": {
    "gradient_descent": {
      "name": "Gradient Descent",
      "convergence_rate": "O(1/k)",
      "stepsize": "1/L",
      "parameters": ["lipschitz_constant", "initial_radius"],
      "description": "Classic gradient descent with fixed stepsize"
    },
    "heavy_ball": {
      "name": "Heavy Ball Method",
      "convergence_rate": "O(1/k)",
      "parameters": ["lipschitz_constant", "initial_radius", "momentum"]
    },
    "accelerated_gradient": {
      "name": "Nesterov Accelerated Gradient",
      "convergence_rate": "O(1/k²)",
      "stepsize": "adaptive"
    }
  },
  "total_algorithms": 3
}
```

### 2. get_default_configs

**Description**: Get default configuration parameters for all algorithms

**Returns:** Default configurations loaded from `configs/*.json` files

### 3. validate_pepflow_installation

**Description**: Validate that PEPFlow framework is properly installed

**Returns:**
```python
{
  "status": "success",
  "pepflow_available": true,
  "dependencies": {
    "pepflow": "available",
    "sympy": "version 1.x.x",
    "numpy": "version 1.x.x",
    "matplotlib": "available"
  },
  "message": "PEPFlow framework is properly installed"
}
```

---

## Workflow Examples

### Quick Analysis (Synchronous)

```python
# 1. Analyze gradient descent with default parameters
result = analyze_gradient_descent(iterations=5, proof_steps=2)

# 2. Check if convergence bound is verified
if result["status"] == "success" and result["convergence_verified"]:
    print("O(1/k) convergence verified!")
    print(f"Final optimal value: {result['numerical_results'][-1]}")

# 3. Compare algorithms
comparison = analyze_optimization_algorithm(
    algorithm="comparison",
    iterations=5
)
```

### Background Analysis (Submit API)

```python
# 1. Submit large-scale analysis
job = submit_gradient_descent_analysis(
    iterations=50,
    proof_steps=20,
    job_name="large_scale_gd"
)
job_id = job["job_id"]

# 2. Monitor progress
while True:
    status = get_job_status(job_id)
    print(f"Status: {status['status']}")

    if status["status"] == "completed":
        break
    elif status["status"] == "failed":
        print(f"Job failed: {status['error']}")
        break

    time.sleep(10)  # Check every 10 seconds

# 3. Get results
result = get_job_result(job_id)
if result["status"] == "success":
    analysis = result["result"]
    print("Analysis completed!")
```

### Parameter Sweep Example

```python
# Submit parameter sweep for momentum optimization
sweep = submit_parameter_sweep(
    algorithm="heavy_ball",
    parameter_ranges={
        "momentum": [0.1, 0.3, 0.5, 0.7, 0.9],
        "lipschitz_constant": [0.5, 1.0, 2.0]
    },
    base_iterations=15,
    job_name="momentum_optimization"
)

# Monitor and get results...
```

---

## Configuration Files

The server uses configuration files from `configs/` directory:

### gradient_descent_config.json
```json
{
  "_description": "Gradient descent convergence analysis configuration",
  "algorithm": {"name": "gradient_descent", "stepsize": "1/L"},
  "analysis": {"iterations": 8, "proof_steps": 2},
  "parameters": {"lipschitz_constant": 1.0, "initial_radius": 1.0},
  "output": {"save_plot": true, "output_dir": "results/gradient_descent"},
  "numerical_settings": {"solver": "OSQP", "precision": 1e-10}
}
```

### accelerated_gradient_config.json
```json
{
  "_description": "Nesterov AGM analysis configuration",
  "algorithm": {"name": "accelerated_gradient", "theta_computation": true},
  "analysis": {"iterations": 5, "proof_steps": 3},
  "parameters": {"lipschitz_constant": 1.0, "initial_radius": 1.0},
  "output": {"save_plot": true, "dual_plots": true}
}
```

---

## Error Handling

All tools return structured error responses:

```python
{
  "status": "error",
  "error": "PEPFlow framework not available: ModuleNotFoundError",
  "installation_command": "mamba run -p ./env pip install -e repo/PEPFlow"
}
```

**Common Error Types:**
1. **ImportError**: PEPFlow framework not installed
2. **ValueError**: Invalid algorithm name or parameters
3. **FileNotFoundError**: Config file not found
4. **RuntimeError**: Solver failures or convergence issues

---

## Performance Considerations

### Runtime Guidelines

| Iterations | Proof Steps | Est. Runtime | Recommended API |
|-----------|-------------|--------------|-----------------|
| 1-5 | 1-3 | 5-20 seconds | Sync |
| 6-10 | 4-5 | 20-60 seconds | Sync |
| 11-20 | 6-10 | 1-5 minutes | Submit |
| 20+ | 10+ | 5+ minutes | Submit |

### Memory Usage
- **Baseline**: ~50-100 MB per analysis
- **Theta computation**: +50 MB for AGM
- **Comparison mode**: 3x baseline for 3 algorithms
- **Parameter sweep**: Linear scaling with parameter combinations

### Optimization Tips
1. **Use skip_plot=True** for faster execution in scripts
2. **Batch parameter sweeps** instead of individual jobs
3. **Monitor job logs** for debugging failed runs
4. **Use appropriate iteration counts** for your analysis needs

---

## Installation and Setup

### Prerequisites
```bash
# Activate environment
mamba activate ./env  # or: conda activate ./env

# Install MCP dependencies (already done in Step 6)
pip install fastmcp loguru
```

### Starting the Server
```bash
# Development mode (with inspector)
mamba run -p ./env fastmcp dev src/server.py

# Production mode
mamba run -p ./env python src/server.py
```

### Verify Installation
```python
# Test PEPFlow availability
result = validate_pepflow_installation()
print(result["message"])

# List available algorithms
algorithms = get_available_algorithms()
print(f"Available: {list(algorithms['algorithms'].keys())}")
```

---

## Tool Summary

| Category | Tool Count | Description |
|----------|------------|-------------|
| **Job Management** | 5 | Status, result, log, cancel, list |
| **Sync Analysis** | 3 | Fast optimization analysis |
| **Submit Analysis** | 4 | Background/batch processing |
| **Utilities** | 3 | Algorithm info, configs, validation |
| **Total** | **15** | Complete optimization analysis toolkit |

---

## Success Criteria Verification

- ✅ MCP server created at `src/server.py`
- ✅ Job manager implemented for async operations (`src/jobs/manager.py`)
- ✅ Sync tools created for fast operations (<2 minutes)
- ✅ Submit tools created for long-running operations (>2 minutes)
- ✅ Batch processing support for parameter sweeps
- ✅ Job management tools working (status, result, log, cancel, list)
- ✅ All tools have clear descriptions for LLM use
- ✅ Error handling returns structured responses
- ✅ Server starts without errors: `mamba run -p ./env python src/server.py`
- ✅ Core functionality tested and verified working
- ✅ Documentation complete with examples and workflows

## Final Notes

**Ready for Production**: The MCP server is fully functional and ready to provide optimization analysis tools to LLM-based applications. All core functionality has been tested, and the server provides both quick interactive analysis and background processing for computationally intensive tasks.

**Next Steps**: Deploy the server and integrate with LLM applications that need optimization analysis capabilities using the PEPFlow framework.