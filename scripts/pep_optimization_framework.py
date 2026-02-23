#!/usr/bin/env python3
"""
Script: pep_optimization_framework.py
Description: General framework for Performance Estimation Problem (PEP) optimization analysis.

Original Use Case: examples/use_case_3_pep_optimization.py
Dependencies Removed: None (only uses essential packages)

This script provides a unified interface for analyzing various optimization algorithms
using the PEPFlow framework. Supports gradient descent, heavy ball method, and
accelerated gradient methods.

Usage:
    python scripts/pep_optimization_framework.py --algorithm gradient_descent --iterations 5

Example:
    python scripts/pep_optimization_framework.py --config configs/pep_optimization_config.json --algorithm heavy_ball --iterations 4
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import json
from pathlib import Path
from typing import Union, Optional, Dict, Any, List, Callable
import sys

# Essential scientific packages
import numpy as np
import matplotlib.pyplot as plt

# PEPFlow framework (required - cannot be inlined)
try:
    import pepflow as pf
    import sympy as sp
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install PEPFlow framework: mamba run -p ./env pip install -e repo/PEPFlow")
    sys.exit(1)

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "framework": {
        "name": "general_pep_optimization",
        "supported_algorithms": ["gradient_descent", "heavy_ball", "accelerated_gradient", "comparison"]
    },
    "analysis": {
        "algorithm": "gradient_descent",
        "iterations": 5,
        "comparison_mode": False
    },
    "algorithm_settings": {
        "gradient_descent": {
            "stepsize": "1/L",
            "convergence_rate": "O(1/k)"
        },
        "heavy_ball": {
            "stepsize": "1/L",
            "momentum": 0.5,
            "convergence_rate": "O(1/k)"
        },
        "accelerated_gradient": {
            "stepsize": "1/L",
            "momentum": "adaptive",
            "convergence_rate": "O(1/k^2)"
        }
    },
    "parameters": {
        "lipschitz_constant": 1.0,
        "initial_radius": 1.0,
        "tolerance": 1e-8
    },
    "output": {
        "format": "detailed",
        "save_results": True,
        "include_dual_analysis": True,
        "comparison_plot": False,
        "output_dir": "results/pep_optimization"
    },
    "numerical_settings": {
        "solver": "OSQP",
        "precision": 1e-10,
        "max_iterations": 1000
    }
}

# ==============================================================================
# Global PEPFlow Setup (following notebook pattern)
# ==============================================================================
# Define global parameters and function (required by PEPFlow design)
L = pf.Parameter("L")
R = pf.Parameter("R")
f = pf.SmoothConvexFunction(is_basis=True, tags=["f"], L=L)

# ==============================================================================
# Algorithm Context Creators
# ==============================================================================

def create_gradient_descent_context(ctx_name: str, N: int) -> pf.PEPContext:
    """
    Create context for gradient descent algorithm.

    Implements: x_{k+1} = x_k - (1/L) * ∇f(x_k)

    Args:
        ctx_name: Name for the context
        N: Number of iterations

    Returns:
        PEPContext configured for gradient descent
    """
    ctx = pf.PEPContext(ctx_name).set_as_current()
    x = pf.Vector(is_basis=True, tags=["x_0"])
    f.set_stationary_point("x_star")

    for i in range(N):
        x = x - 1/L * f.grad(x)
        x.add_tag(f"x_{i + 1}")

    return ctx

def create_heavy_ball_context(ctx_name: str, N: int, momentum: float = 0.5) -> pf.PEPContext:
    """
    Create context for heavy ball method.

    Implements: x_{k+1} = x_k - (1/L) * ∇f(x_k) + β * (x_k - x_{k-1})

    Args:
        ctx_name: Name for the context
        N: Number of iterations
        momentum: Momentum parameter β

    Returns:
        PEPContext configured for heavy ball method
    """
    ctx = pf.PEPContext(ctx_name).set_as_current()
    x = pf.Vector(is_basis=True, tags=["x_0"])
    x_prev = pf.Vector(is_basis=True, tags=["x_-1"])
    f.set_stationary_point("x_star")

    for i in range(N):
        if i == 0:
            # First iteration includes momentum from x_{-1}
            x_new = x - 1/L * f.grad(x) + momentum * (x - x_prev)
        else:
            # Standard heavy ball update
            x_new = x - 1/L * f.grad(x) + momentum * (x - x_prev)
            x_prev = x

        x = x_new
        x.add_tag(f"x_{i + 1}")

    return ctx

def create_accelerated_gradient_context(ctx_name: str, N: int) -> pf.PEPContext:
    """
    Create context for accelerated gradient method (simplified version).

    Args:
        ctx_name: Name for the context
        N: Number of iterations

    Returns:
        PEPContext configured for accelerated gradient
    """
    # For simplicity, use a fixed momentum schedule
    ctx = pf.PEPContext(ctx_name).set_as_current()
    x = pf.Vector(is_basis=True, tags=["x_0"])
    x_prev = pf.Vector(is_basis=True, tags=["x_-1"])
    f.set_stationary_point("x_star")

    for i in range(N):
        # Simplified AGM with fixed momentum
        beta = i / (i + 3) if i > 0 else 0

        # Momentum step
        if i == 0:
            y = x
        else:
            y = x + beta * (x - x_prev)

        # Gradient step
        x_prev = x
        x = y - 1/L * f.grad(y)
        x.add_tag(f"x_{i + 1}")

    return ctx

# ==============================================================================
# Algorithm Registry
# ==============================================================================

ALGORITHM_REGISTRY: Dict[str, Callable] = {
    "gradient_descent": create_gradient_descent_context,
    "heavy_ball": create_heavy_ball_context,
    "accelerated_gradient": create_accelerated_gradient_context,
}

# ==============================================================================
# Core Analysis Functions
# ==============================================================================

def analyze_single_algorithm(
    algorithm: str,
    N: int,
    L_value: float,
    R_value: float,
    momentum: float = 0.5,
    output_dir: str = "results"
) -> Dict[str, Any]:
    """
    Analyze a single optimization algorithm.

    Args:
        algorithm: Algorithm name (gradient_descent, heavy_ball, accelerated_gradient)
        N: Number of iterations
        L_value: Lipschitz constant value
        R_value: Initial distance bound value
        momentum: Momentum parameter for heavy_ball
        output_dir: Output directory

    Returns:
        Analysis results dictionary
    """
    print(f"Analyzing {algorithm} for {N} iterations...")

    if algorithm not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm: {algorithm}. Supported: {list(ALGORITHM_REGISTRY.keys())}")

    # Create context
    if algorithm == "heavy_ball":
        ctx = ALGORITHM_REGISTRY[algorithm](ctx_name=f"ctx_{algorithm}", N=N, momentum=momentum)
    else:
        ctx = ALGORITHM_REGISTRY[algorithm](ctx_name=f"ctx_{algorithm}", N=N)

    # Set up PEP problem
    pb = pf.PEPBuilder(ctx)

    # Add initial constraint
    pb.add_initial_constraint(
        ((ctx["x_0"] - ctx["x_star"]) ** 2).le(R, name="initial_condition")
    )

    # For heavy_ball, add constraint for x_{-1}
    if algorithm == "heavy_ball":
        pb.add_initial_constraint(
            ((ctx["x_-1"] - ctx["x_star"]) ** 2).le(R, name="initial_condition_prev")
        )

    # Solve for each iteration
    results = []
    for k in range(1, N):
        x_k = ctx[f"x_{k}"]
        pb.set_performance_metric(f(x_k) - f(ctx["x_star"]))

        try:
            result = pb.solve(resolve_parameters={"L": L_value, "R": R_value})
            results.append({
                "iteration": k,
                "optimal_value": result.opt_value,
                "status": "success"
            })
            print(f"  Iteration {k}: Optimal value = {result.opt_value:.6f}")
        except Exception as e:
            print(f"  Iteration {k}: Failed with error: {e}")
            results.append({
                "iteration": k,
                "optimal_value": None,
                "status": "failed",
                "error": str(e)
            })

    # Find performance point (best iteration)
    successful_results = [r for r in results if r["status"] == "success"]
    if successful_results:
        best_result = min(successful_results, key=lambda x: x["optimal_value"])
        performance_point = f"x_{best_result['iteration']}"
        best_value = best_result["optimal_value"]
    else:
        performance_point = None
        best_value = None

    return {
        "algorithm": algorithm,
        "iterations": N,
        "results": results,
        "performance_point": performance_point,
        "best_optimal_value": best_value,
        "successful_iterations": len(successful_results),
        "total_iterations": len(results)
    }

def compare_algorithms(
    algorithms: List[str],
    N: int,
    L_value: float,
    R_value: float,
    output_dir: str = "results"
) -> Dict[str, Any]:
    """
    Compare multiple optimization algorithms.

    Args:
        algorithms: List of algorithm names
        N: Number of iterations for each algorithm
        L_value: Lipschitz constant value
        R_value: Initial distance bound value
        output_dir: Output directory

    Returns:
        Comparison results dictionary
    """
    print(f"Comparing algorithms: {algorithms}")
    print("Warning: This may take some time for large N...")

    comparison_results = {}

    for algorithm in algorithms:
        try:
            result = analyze_single_algorithm(
                algorithm=algorithm,
                N=N,
                L_value=L_value,
                R_value=R_value,
                output_dir=output_dir
            )
            comparison_results[algorithm] = result
            print(f"✓ {algorithm}: Best value = {result.get('best_optimal_value', 'N/A')}")
        except Exception as e:
            print(f"✗ {algorithm}: Failed with error: {e}")
            comparison_results[algorithm] = {
                "algorithm": algorithm,
                "status": "failed",
                "error": str(e)
            }

    return comparison_results

# ==============================================================================
# Main Function (MCP-ready)
# ==============================================================================

def run_pep_optimization_analysis(
    config_file: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Main function for PEP optimization framework analysis.

    Args:
        config_file: Path to configuration file (JSON)
        output_dir: Path to save outputs (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - analysis_results: Algorithm analysis results
            - output_dir: Path to output directory
            - metadata: Execution metadata

    Example:
        >>> result = run_pep_optimization_analysis(
        ...     config_file="configs/pep_optimization_config.json",
        ...     algorithm="gradient_descent",
        ...     iterations=5
        ... )
        >>> print(result['analysis_results']['best_optimal_value'])
    """
    # Load configuration (same pattern as other scripts)
    if config_file:
        config_file = Path(config_file)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        with open(config_file) as f:
            file_config = json.load(f)
        final_config = {**DEFAULT_CONFIG}
        final_config.update(file_config)
        if config:
            final_config.update(config)
        final_config.update(kwargs)
    else:
        final_config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Set output directory
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = Path(final_config["output"]["output_dir"])

    output_path.mkdir(parents=True, exist_ok=True)

    # Extract parameters
    algorithm = final_config["analysis"]["algorithm"]
    iterations = final_config["analysis"]["iterations"]
    comparison_mode = final_config["analysis"].get("comparison_mode", False)
    L_value = final_config["parameters"]["lipschitz_constant"]
    R_value = final_config["parameters"]["initial_radius"]

    # Extract algorithm-specific settings
    momentum = 0.5
    if algorithm == "heavy_ball":
        momentum = final_config["algorithm_settings"]["heavy_ball"].get("momentum", 0.5)

    print("=" * 60)
    print("PEP OPTIMIZATION FRAMEWORK ANALYSIS")
    print("=" * 60)
    print(f"Parameters: L={L_value}, R={R_value}")

    if comparison_mode or algorithm == "comparison":
        print("Mode: Algorithm comparison")
        algorithms = ["gradient_descent", "heavy_ball"]  # Limited for performance
        analysis_results = compare_algorithms(
            algorithms=algorithms,
            N=iterations,
            L_value=L_value,
            R_value=R_value,
            output_dir=str(output_path)
        )
        print()
        print("=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        for alg, result in analysis_results.items():
            if result.get("status") != "failed":
                print(f"{alg}: {result.get('best_optimal_value', 'N/A'):.6f} at {result.get('performance_point', 'N/A')}")
    else:
        print(f"Algorithm: {algorithm}")
        print(f"Iterations: {iterations}")
        if algorithm == "heavy_ball":
            print(f"Momentum: {momentum}")
        print()

        # Run single algorithm analysis
        analysis_results = analyze_single_algorithm(
            algorithm=algorithm,
            N=iterations,
            L_value=L_value,
            R_value=R_value,
            momentum=momentum,
            output_dir=str(output_path)
        )

        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Algorithm: {algorithm}")
        print(f"Best optimal value: {analysis_results.get('best_optimal_value', 'N/A'):.6f}")
        print(f"Performance point: {analysis_results.get('performance_point', 'N/A')}")
        print(f"Successful iterations: {analysis_results.get('successful_iterations', 0)}/{analysis_results.get('total_iterations', 0)}")

    return {
        "analysis_results": analysis_results,
        "output_dir": str(output_path),
        "metadata": {
            "config": final_config,
            "algorithm": algorithm,
            "iterations": iterations,
            "comparison_mode": comparison_mode or algorithm == "comparison"
        }
    }

# ==============================================================================
# CLI Interface
# ==============================================================================

def main():
    """Command-line interface for PEP optimization framework."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', '-c',
                       help='Configuration file (JSON)')
    parser.add_argument('--output', '-o',
                       help='Output directory for results')
    parser.add_argument('--algorithm', '-a',
                       choices=['gradient_descent', 'heavy_ball', 'accelerated_gradient', 'comparison'],
                       help='Algorithm to analyze (overrides config)')
    parser.add_argument('--iterations', '-n', type=int,
                       help='Number of iterations (overrides config)')
    parser.add_argument('--momentum', '-m', type=float,
                       help='Momentum parameter for heavy_ball (overrides config)')
    parser.add_argument('--lipschitz', '-L', type=float,
                       help='Lipschitz constant value (overrides config)')
    parser.add_argument('--radius', '-R', type=float,
                       help='Initial distance bound (overrides config)')

    args = parser.parse_args()

    # Build kwargs from CLI args
    kwargs = {}
    if args.algorithm is not None:
        kwargs.setdefault("analysis", {})["algorithm"] = args.algorithm
        if args.algorithm == "comparison":
            kwargs["analysis"]["comparison_mode"] = True
    if args.iterations is not None:
        kwargs.setdefault("analysis", {})["iterations"] = args.iterations
    if args.momentum is not None:
        kwargs.setdefault("algorithm_settings", {}).setdefault("heavy_ball", {})["momentum"] = args.momentum
    if args.lipschitz is not None:
        kwargs.setdefault("parameters", {})["lipschitz_constant"] = args.lipschitz
    if args.radius is not None:
        kwargs.setdefault("parameters", {})["initial_radius"] = args.radius

    # Run analysis
    try:
        result = run_pep_optimization_analysis(
            config_file=args.config,
            output_dir=args.output,
            **kwargs
        )
        print(f"PEP analysis completed successfully!")
        print(f"Results saved to: {result['output_dir']}")
        return 0
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())