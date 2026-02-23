#!/usr/bin/env python3
"""
Use Case 3: General PEP (Performance Estimation Problem) Optimization

This script provides a general interface for setting up and solving Performance
Estimation Problems for various optimization algorithms. It demonstrates the core
PEPFlow workflow: define algorithm, set up constraints, solve PEP, and analyze results.

Author: Extracted from PEPFlow examples
License: Apache 2.0
"""

import argparse
import pepflow as pf
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Callable

# Define global parameters and function (following the notebook pattern)
L = pf.Parameter("L")
R = pf.Parameter("R")
f = pf.SmoothConvexFunction(is_basis=True, tags=["f"], L=L)


def make_gradient_descent_context(ctx_name: str, N: int) -> pf.PEPContext:
    """Create context for gradient descent."""
    ctx = pf.PEPContext(ctx_name).set_as_current()
    x = pf.Vector(is_basis=True, tags=["x_0"])
    f.set_stationary_point("x_star")
    for i in range(N):
        x = x - 1/L * f.grad(x)
        x.add_tag(f"x_{i + 1}")
    return ctx


def make_heavy_ball_context(ctx_name: str, N: int, momentum: float = 0.5) -> pf.PEPContext:
    """Create context for heavy ball method."""
    ctx = pf.PEPContext(ctx_name).set_as_current()
    x = pf.Vector(is_basis=True, tags=["x_0"])
    x_prev = pf.Vector(is_basis=True, tags=["x_-1"])
    f.set_stationary_point("x_star")

    for i in range(N):
        if i == 0:
            # First iteration: x_1 = x_0 - (1/L) * ∇f(x_0) + β * (x_0 - x_{-1})
            x_new = x - 1/L * f.grad(x) + momentum * (x - x_prev)
        else:
            # Standard heavy ball update
            x_new = x - 1/L * f.grad(x) + momentum * (x - x_prev)

        x_prev = x
        x = x_new
        x.add_tag(f"x_{i + 1}")

    return ctx


def solve_pep_problem(ctx: pf.PEPContext, L_value: float = 1.0, R_value: float = 1.0,
                     performance_point: str = None) -> dict:
    """
    Solve a PEP problem for given context.

    Args:
        ctx: PEPContext with algorithm setup
        L_value: Lipschitz constant value
        R_value: Initial distance bound
        performance_point: Point to evaluate performance at (default: last iterate)

    Returns:
        Dictionary with solution results
    """
    pb = pf.PEPBuilder(ctx)

    # Add initial condition
    pb.add_initial_constraint(
        ((ctx["x_0"] - ctx["x_star"]) ** 2).le(R, name="initial_condition")
    )

    # Find performance point if not specified
    if performance_point is None:
        # Try to find the highest numbered x iterate
        max_idx = 0
        for vector in ctx.vectors:
            for tag in vector.tags:
                if tag.startswith("x_") and tag != "x_star":
                    try:
                        idx = int(tag.split("_")[1])
                        if idx >= 0:  # Only positive indices
                            max_idx = max(max_idx, idx)
                    except (IndexError, ValueError):
                        continue
        performance_point = f"x_{max_idx}"

    # Set performance metric
    x_final = ctx[performance_point]
    pb.set_performance_metric(f(x_final) - f(ctx["x_star"]))

    # Solve
    result = pb.solve(resolve_parameters={"L": L_value, "R": R_value})

    return {
        "optimal_value": result.opt_value,
        "performance_point": performance_point,
        "context_name": ctx.name,
        "result_object": result
    }


def compare_algorithms(algorithms: List[str], max_iterations: int = 5,
                      L_value: float = 1.0, R_value: float = 1.0,
                      momentum: float = 0.5) -> Dict[str, List[float]]:
    """
    Compare multiple algorithms across different iteration counts.

    Args:
        algorithms: List of algorithm names to compare
        max_iterations: Maximum number of iterations to test
        L_value: Lipschitz constant value
        R_value: Initial distance bound
        momentum: Momentum parameter for heavy ball

    Returns:
        Dictionary mapping algorithm names to lists of optimal values
    """
    results = {}

    for alg_name in algorithms:
        print(f"\nAnalyzing {alg_name}:")
        alg_results = []

        for N in range(1, max_iterations + 1):
            try:
                if alg_name == "gradient_descent":
                    ctx = make_gradient_descent_context(f"ctx_{alg_name}_{N}", N)
                elif alg_name == "heavy_ball":
                    ctx = make_heavy_ball_context(f"ctx_{alg_name}_{N}", N, momentum)
                else:
                    raise ValueError(f"Unknown algorithm: {alg_name}")

                result = solve_pep_problem(ctx, L_value, R_value)
                alg_results.append(result["optimal_value"])
                print(f"  Iteration {N}: {result['optimal_value']:.6f}")

            except Exception as e:
                print(f"  Iteration {N}: Error - {e}")
                alg_results.append(float('inf'))

        results[alg_name] = alg_results

    return results


def analyze_single_algorithm(algorithm: str, iterations: int, L_value: float = 1.0,
                           R_value: float = 1.0, momentum: float = 0.5) -> dict:
    """
    Analyze a single algorithm for specified iterations.

    Args:
        algorithm: Algorithm name
        iterations: Number of iterations
        L_value: Lipschitz constant value
        R_value: Initial distance bound
        momentum: Momentum parameter for heavy ball

    Returns:
        Analysis results
    """
    print(f"Analyzing {algorithm} for {iterations} iterations...")

    if algorithm == "gradient_descent":
        ctx = make_gradient_descent_context(f"ctx_{algorithm}", iterations)
    elif algorithm == "heavy_ball":
        ctx = make_heavy_ball_context(f"ctx_{algorithm}", iterations, momentum)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    result = solve_pep_problem(ctx, L_value, R_value)

    print(f"Optimal value: {result['optimal_value']:.10f}")
    print(f"Performance point: {result['performance_point']}")

    return result


def plot_comparison(results: Dict[str, List[float]], output_dir: str = "results") -> None:
    """Plot comparison of algorithm results."""
    plt.figure(figsize=(10, 6))

    for alg_name, values in results.items():
        iters = range(1, len(values) + 1)
        plt.plot(iters, values, 'o-', label=alg_name, linewidth=2, markersize=6)

    plt.xlabel("Iteration")
    plt.ylabel("Optimal Value f(x_k) - f(x*)")
    plt.title("Algorithm Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    output_path = Path(output_dir) / "algorithm_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")

    plt.show()


def main():
    """Main function for PEP optimization analysis."""
    parser = argparse.ArgumentParser(description="General PEP Optimization Analysis")
    parser.add_argument("--algorithm", "-a",
                       choices=["gradient_descent", "heavy_ball", "comparison"],
                       default="comparison",
                       help="Algorithm to analyze or 'comparison' for multi-algorithm comparison")
    parser.add_argument("--iterations", "-n", type=int, default=6,
                       help="Number of iterations for analysis")
    parser.add_argument("--lipschitz", "-L", type=float, default=1.0,
                       help="Lipschitz constant value")
    parser.add_argument("--radius", "-R", type=float, default=1.0,
                       help="Initial distance bound")
    parser.add_argument("--momentum", "-m", type=float, default=0.5,
                       help="Momentum parameter for Heavy Ball method")
    parser.add_argument("--output-dir", "-o", type=str, default="results/uc_003",
                       help="Output directory for plots and data")
    parser.add_argument("--analyze-duals", action="store_true",
                       help="Analyze dual variables (advanced)")

    args = parser.parse_args()

    print("=" * 60)
    print("GENERAL PEP OPTIMIZATION ANALYSIS")
    print("=" * 60)
    print(f"Parameters: L={args.lipschitz}, R={args.radius}")
    print(f"Algorithm: {args.algorithm}")
    print()

    try:
        if args.algorithm == "comparison":
            algorithms = ['gradient_descent', 'heavy_ball']
            print(f"Comparing algorithms: {algorithms}")
            print(f"Testing iterations: {list(range(1, args.iterations + 1))}")

            results = compare_algorithms(
                algorithms=algorithms,
                max_iterations=args.iterations,
                L_value=args.lipschitz,
                R_value=args.radius,
                momentum=args.momentum
            )

            print("\n" + "=" * 60)
            print("COMPARISON SUMMARY")
            print("=" * 60)
            for alg_name, values in results.items():
                print(f"{alg_name}: {[f'{v:.6f}' for v in values]}")

            plot_comparison(results, args.output_dir)

        else:
            # Single algorithm analysis
            result = analyze_single_algorithm(
                algorithm=args.algorithm,
                iterations=args.iterations,
                L_value=args.lipschitz,
                R_value=args.radius,
                momentum=args.momentum
            )

            if args.analyze_duals:
                print("\nDual variable analysis:")
                print(f"Context name: {result['context_name']}")
                print("(Dual analysis would be implemented here)")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())