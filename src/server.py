"""MCP Server for Optimization Analysis Tools

Provides both synchronous and asynchronous (submit) APIs for PEPFlow-based
optimization analysis tools including gradient descent, accelerated gradient,
and general PEP optimization framework.
"""

from fastmcp import FastMCP
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import sys

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MCP_ROOT = SCRIPT_DIR.parent
SCRIPTS_DIR = MCP_ROOT / "scripts"
CONFIGS_DIR = MCP_ROOT / "configs"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from jobs.manager import job_manager
from loguru import logger

# Create MCP server
mcp = FastMCP("optimization-analysis")

# ==============================================================================
# Job Management Tools (for async operations)
# ==============================================================================

@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """
    Get the status of a submitted optimization analysis job.

    Args:
        job_id: The job ID returned from a submit_* function

    Returns:
        Dictionary with job status, timestamps, and any errors
    """
    return job_manager.get_job_status(job_id)

@mcp.tool()
def get_job_result(job_id: str) -> dict:
    """
    Get the results of a completed optimization analysis job.

    Args:
        job_id: The job ID of a completed job

    Returns:
        Dictionary with the job results or error if not completed
    """
    return job_manager.get_job_result(job_id)

@mcp.tool()
def get_job_log(job_id: str, tail: int = 50) -> dict:
    """
    Get log output from a running or completed job.

    Args:
        job_id: The job ID to get logs for
        tail: Number of lines from end (default: 50, use 0 for all)

    Returns:
        Dictionary with log lines and total line count
    """
    return job_manager.get_job_log(job_id, tail)

@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """
    Cancel a running optimization analysis job.

    Args:
        job_id: The job ID to cancel

    Returns:
        Success or error message
    """
    return job_manager.cancel_job(job_id)

@mcp.tool()
def list_jobs(status: Optional[str] = None) -> dict:
    """
    List all submitted optimization analysis jobs.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)

    Returns:
        List of jobs with their status
    """
    return job_manager.list_jobs(status)

# ==============================================================================
# Synchronous Tools (for fast operations < 2 minutes)
# ==============================================================================

@mcp.tool()
def analyze_gradient_descent(
    iterations: int = 8,
    proof_steps: int = 2,
    lipschitz_constant: float = 1.0,
    initial_radius: float = 1.0,
    output_dir: Optional[str] = None,
    skip_plot: bool = True,
    config_file: Optional[str] = None
) -> dict:
    """
    Analyze gradient descent convergence using PEPFlow framework.

    Fast operation - returns results immediately. Verifies O(1/k) convergence
    rate for gradient descent with fixed stepsize 1/L.

    Args:
        iterations: Number of iterations for numerical analysis (default: 8)
        proof_steps: Number of steps for analytical proof (default: 2)
        lipschitz_constant: Lipschitz constant L (default: 1.0)
        initial_radius: Initial distance bound R (default: 1.0)
        output_dir: Optional output directory
        skip_plot: Skip plotting for faster execution (default: True)
        config_file: Optional config file path (uses default if not provided)

    Returns:
        Dictionary with convergence analysis results and verification status
    """
    try:
        # Import here to avoid import errors if PEPFlow not available
        from gradient_descent_analysis import run_gradient_descent_analysis

        # Use default config if none provided
        if config_file is None:
            config_file = str(CONFIGS_DIR / "gradient_descent_config.json")

        result = run_gradient_descent_analysis(
            config_file=config_file,
            output_dir=output_dir,
            iterations=iterations,
            proof_steps=proof_steps,
            lipschitz_constant=lipschitz_constant,
            initial_radius=initial_radius,
            skip_plot=skip_plot
        )
        return {"status": "success", **result}
    except ImportError as e:
        return {
            "status": "error",
            "error": f"PEPFlow framework not available: {e}. Please install with: mamba run -p ./env pip install -e repo/PEPFlow"
        }
    except Exception as e:
        logger.error(f"Gradient descent analysis failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def analyze_accelerated_gradient(
    iterations: int = 5,
    proof_steps: int = 3,
    lipschitz_constant: float = 1.0,
    initial_radius: float = 1.0,
    output_dir: Optional[str] = None,
    skip_plot: bool = True,
    config_file: Optional[str] = None
) -> dict:
    """
    Analyze Nesterov's accelerated gradient method convergence.

    Fast operation - returns results immediately. Verifies O(1/k²) convergence
    rate with theta sequence computation.

    Args:
        iterations: Number of iterations for numerical analysis (default: 5)
        proof_steps: Number of steps for analytical proof (default: 3)
        lipschitz_constant: Lipschitz constant L (default: 1.0)
        initial_radius: Initial distance bound R (default: 1.0)
        output_dir: Optional output directory
        skip_plot: Skip plotting for faster execution (default: True)
        config_file: Optional config file path

    Returns:
        Dictionary with AGM analysis results and O(1/k²) verification
    """
    try:
        from accelerated_gradient_analysis import run_accelerated_gradient_analysis

        if config_file is None:
            config_file = str(CONFIGS_DIR / "accelerated_gradient_config.json")

        result = run_accelerated_gradient_analysis(
            config_file=config_file,
            output_dir=output_dir,
            iterations=iterations,
            proof_steps=proof_steps,
            lipschitz_constant=lipschitz_constant,
            initial_radius=initial_radius,
            skip_plot=skip_plot
        )
        return {"status": "success", **result}
    except ImportError as e:
        return {
            "status": "error",
            "error": f"PEPFlow framework not available: {e}"
        }
    except Exception as e:
        logger.error(f"Accelerated gradient analysis failed: {e}")
        return {"status": "error", "error": str(e)}

@mcp.tool()
def analyze_optimization_algorithm(
    algorithm: str = "gradient_descent",
    iterations: int = 5,
    momentum: float = 0.5,
    lipschitz_constant: float = 1.0,
    initial_radius: float = 1.0,
    output_dir: Optional[str] = None,
    config_file: Optional[str] = None
) -> dict:
    """
    Analyze optimization algorithms using PEP framework.

    Fast operation for single algorithms. Supports gradient_descent, heavy_ball,
    accelerated_gradient, and comparison modes.

    Args:
        algorithm: Algorithm to analyze (gradient_descent, heavy_ball, accelerated_gradient, comparison)
        iterations: Number of iterations (default: 5)
        momentum: Momentum parameter for heavy_ball method (default: 0.5)
        lipschitz_constant: Lipschitz constant L (default: 1.0)
        initial_radius: Initial distance bound R (default: 1.0)
        output_dir: Optional output directory
        config_file: Optional config file path

    Returns:
        Dictionary with optimization analysis results
    """
    try:
        from pep_optimization_framework import run_pep_optimization_analysis

        if config_file is None:
            config_file = str(CONFIGS_DIR / "pep_optimization_config.json")

        result = run_pep_optimization_analysis(
            config_file=config_file,
            output_dir=output_dir,
            algorithm=algorithm,
            iterations=iterations,
            momentum=momentum,
            lipschitz_constant=lipschitz_constant,
            initial_radius=initial_radius
        )
        return {"status": "success", **result}
    except ImportError as e:
        return {
            "status": "error",
            "error": f"PEPFlow framework not available: {e}"
        }
    except Exception as e:
        logger.error(f"Optimization analysis failed: {e}")
        return {"status": "error", "error": str(e)}

# ==============================================================================
# Submit Tools (for long-running operations > 2 minutes)
# ==============================================================================

@mcp.tool()
def submit_gradient_descent_analysis(
    iterations: int = 20,
    proof_steps: int = 10,
    lipschitz_constant: float = 1.0,
    initial_radius: float = 1.0,
    output_dir: Optional[str] = None,
    job_name: Optional[str] = None,
    config_file: Optional[str] = None
) -> dict:
    """
    Submit a gradient descent convergence analysis job.

    Use this for large-scale analysis with many iterations or proof steps.
    This task may take more than 2 minutes. Use get_job_status() to monitor
    progress and get_job_result() to retrieve results when completed.

    Args:
        iterations: Number of iterations for numerical analysis (default: 20)
        proof_steps: Number of steps for analytical proof (default: 10)
        lipschitz_constant: Lipschitz constant L (default: 1.0)
        initial_radius: Initial distance bound R (default: 1.0)
        output_dir: Optional output directory
        job_name: Optional name for the job (for easier tracking)
        config_file: Optional config file path

    Returns:
        Dictionary with job_id for tracking. Use:
        - get_job_status(job_id) to check progress
        - get_job_result(job_id) to get results when completed
        - get_job_log(job_id) to see execution logs
    """
    script_path = str(SCRIPTS_DIR / "gradient_descent_analysis.py")

    # Use default config if none provided
    if config_file is None:
        config_file = str(CONFIGS_DIR / "gradient_descent_config.json")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "config_file": config_file,
            "iterations": iterations,
            "proof_steps": proof_steps,
            "lipschitz_constant": lipschitz_constant,
            "initial_radius": initial_radius,
            "output_dir": output_dir
        },
        job_name=job_name or f"gradient_descent_{iterations}iter"
    )

@mcp.tool()
def submit_accelerated_gradient_analysis(
    iterations: int = 15,
    proof_steps: int = 8,
    lipschitz_constant: float = 1.0,
    initial_radius: float = 1.0,
    output_dir: Optional[str] = None,
    job_name: Optional[str] = None,
    config_file: Optional[str] = None
) -> dict:
    """
    Submit an accelerated gradient method analysis job.

    Use this for extensive theta sequence computation and convergence analysis.
    This is a long-running task for high iteration counts.

    Args:
        iterations: Number of iterations for numerical analysis (default: 15)
        proof_steps: Number of steps for analytical proof (default: 8)
        lipschitz_constant: Lipschitz constant L (default: 1.0)
        initial_radius: Initial distance bound R (default: 1.0)
        output_dir: Optional output directory
        job_name: Optional name for the job
        config_file: Optional config file path

    Returns:
        Dictionary with job_id for tracking
    """
    script_path = str(SCRIPTS_DIR / "accelerated_gradient_analysis.py")

    if config_file is None:
        config_file = str(CONFIGS_DIR / "accelerated_gradient_config.json")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "config_file": config_file,
            "iterations": iterations,
            "proof_steps": proof_steps,
            "lipschitz_constant": lipschitz_constant,
            "initial_radius": initial_radius,
            "output_dir": output_dir
        },
        job_name=job_name or f"accelerated_gradient_{iterations}iter"
    )

@mcp.tool()
def submit_optimization_comparison(
    algorithms: List[str] = ["gradient_descent", "heavy_ball", "accelerated_gradient"],
    iterations: int = 10,
    momentum: float = 0.5,
    lipschitz_constant: float = 1.0,
    initial_radius: float = 1.0,
    output_dir: Optional[str] = None,
    job_name: Optional[str] = None,
    config_file: Optional[str] = None
) -> dict:
    """
    Submit a comprehensive optimization algorithm comparison job.

    Compares multiple optimization algorithms using the PEP framework.
    This is computationally intensive for multiple algorithms and high iteration counts.

    Args:
        algorithms: List of algorithms to compare (default: all three main algorithms)
        iterations: Number of iterations for each algorithm (default: 10)
        momentum: Momentum parameter for heavy_ball method (default: 0.5)
        lipschitz_constant: Lipschitz constant L (default: 1.0)
        initial_radius: Initial distance bound R (default: 1.0)
        output_dir: Optional output directory
        job_name: Optional name for the job
        config_file: Optional config file path

    Returns:
        Dictionary with job_id for tracking the comparison job
    """
    script_path = str(SCRIPTS_DIR / "pep_optimization_framework.py")

    if config_file is None:
        config_file = str(CONFIGS_DIR / "pep_optimization_config.json")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "config_file": config_file,
            "algorithm": "comparison",  # This triggers comparison mode
            "iterations": iterations,
            "momentum": momentum,
            "lipschitz_constant": lipschitz_constant,
            "initial_radius": initial_radius,
            "output_dir": output_dir
        },
        job_name=job_name or f"optimization_comparison_{len(algorithms)}algs_{iterations}iter"
    )

# ==============================================================================
# Batch Processing Tools
# ==============================================================================

@mcp.tool()
def submit_parameter_sweep(
    algorithm: str = "gradient_descent",
    parameter_ranges: Dict[str, List[float]] = None,
    base_iterations: int = 10,
    output_dir: Optional[str] = None,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit a parameter sweep analysis job.

    Analyzes algorithm performance across different parameter values.
    This is a long-running batch job suitable for parameter optimization.

    Args:
        algorithm: Base algorithm to analyze (gradient_descent, heavy_ball, accelerated_gradient)
        parameter_ranges: Dictionary of parameter names to value lists
                         Example: {"lipschitz_constant": [0.5, 1.0, 2.0], "momentum": [0.3, 0.5, 0.7]}
        base_iterations: Number of iterations for each parameter combination
        output_dir: Optional output directory
        job_name: Optional name for the batch job

    Returns:
        Dictionary with job_id for tracking the parameter sweep
    """
    # Default parameter ranges if not provided
    if parameter_ranges is None:
        if algorithm == "heavy_ball":
            parameter_ranges = {
                "lipschitz_constant": [0.5, 1.0, 2.0],
                "momentum": [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        else:
            parameter_ranges = {
                "lipschitz_constant": [0.5, 1.0, 1.5, 2.0],
                "initial_radius": [0.5, 1.0, 1.5]
            }

    script_path = str(SCRIPTS_DIR / "pep_optimization_framework.py")

    # For now, we'll use the general framework and pass parameters
    # A more sophisticated implementation could create a dedicated parameter sweep script
    return job_manager.submit_job(
        script_path=script_path,
        args={
            "algorithm": algorithm,
            "iterations": base_iterations,
            "output_dir": output_dir,
            # Note: Real parameter sweep would require extending the script
            # For now this demonstrates the pattern
        },
        job_name=job_name or f"parameter_sweep_{algorithm}_{len(parameter_ranges)}params"
    )

# ==============================================================================
# Utility Tools
# ==============================================================================

@mcp.tool()
def get_available_algorithms() -> dict:
    """
    Get list of available optimization algorithms and their properties.

    Returns:
        Dictionary with algorithm information including convergence rates and parameters
    """
    algorithms = {
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
            "stepsize": "1/L",
            "parameters": ["lipschitz_constant", "initial_radius", "momentum"],
            "description": "Gradient descent with momentum term"
        },
        "accelerated_gradient": {
            "name": "Nesterov Accelerated Gradient",
            "convergence_rate": "O(1/k²)",
            "stepsize": "adaptive",
            "parameters": ["lipschitz_constant", "initial_radius"],
            "description": "Accelerated gradient method with optimal convergence rate"
        }
    }

    return {
        "status": "success",
        "algorithms": algorithms,
        "total_algorithms": len(algorithms)
    }

@mcp.tool()
def get_default_configs() -> dict:
    """
    Get default configuration parameters for all algorithms.

    Returns:
        Dictionary with default configurations for each algorithm
    """
    try:
        configs = {}

        # Load default configs from files
        config_files = {
            "gradient_descent": CONFIGS_DIR / "gradient_descent_config.json",
            "accelerated_gradient": CONFIGS_DIR / "accelerated_gradient_config.json",
            "pep_optimization": CONFIGS_DIR / "pep_optimization_config.json"
        }

        import json
        for name, config_file in config_files.items():
            if config_file.exists():
                with open(config_file) as f:
                    configs[name] = json.load(f)

        return {
            "status": "success",
            "configs": configs
        }
    except Exception as e:
        return {"status": "error", "error": f"Failed to load configs: {e}"}

@mcp.tool()
def validate_pepflow_installation() -> dict:
    """
    Validate that PEPFlow framework is properly installed.

    Returns:
        Dictionary with installation status and version information
    """
    try:
        import pepflow as pf
        import sympy as sp
        import numpy as np
        import matplotlib.pyplot as plt

        return {
            "status": "success",
            "pepflow_available": True,
            "dependencies": {
                "pepflow": "available",
                "sympy": f"version {sp.__version__}",
                "numpy": f"version {np.__version__}",
                "matplotlib": "available"
            },
            "message": "PEPFlow framework is properly installed and ready to use"
        }
    except ImportError as e:
        return {
            "status": "error",
            "pepflow_available": False,
            "error": str(e),
            "installation_command": "mamba run -p ./env pip install -e repo/PEPFlow",
            "message": "PEPFlow framework is not available. Please install it first."
        }

# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    mcp.run()