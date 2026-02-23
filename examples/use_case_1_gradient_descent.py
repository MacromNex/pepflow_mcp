#!/usr/bin/env python3
"""
Use Case 1: Gradient Descent Method Analysis

This script demonstrates the analysis of gradient descent convergence using PEPFlow.
It performs numerical verification and analytical proof of the O(1/k) convergence rate.

Author: Extracted from PEPFlow examples
License: Apache 2.0
"""

import argparse
import pepflow as pf
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import itertools
from pathlib import Path

# Define global parameters and function (following the notebook pattern)
L = pf.Parameter("L")
f = pf.SmoothConvexFunction(is_basis=True, tags=["f"], L=L)


def make_ctx_gd(ctx_name: str, N: int, stepsize) -> pf.PEPContext:
    """
    Create PEPContext for gradient descent analysis.
    Based on the original notebook implementation.

    Args:
        ctx_name: Name for the context
        N: Number of iterations
        stepsize: Step size parameter (typically 1/L)

    Returns:
        PEPContext configured for gradient descent
    """
    ctx_gd = pf.PEPContext(ctx_name).set_as_current()
    x = pf.Vector(is_basis=True, tags=["x_0"])
    f.set_stationary_point("x_star")
    for i in range(N):
        x = x - stepsize * f.grad(x)
        x.add_tag(f"x_{i + 1}")
    return ctx_gd


def analyze_numerical_convergence(N: int = 8, L_value: float = 1.0, R_value: float = 1.0,
                                save_plot: bool = True, output_dir: str = "examples/data") -> list:
    """
    Perform numerical analysis of gradient descent convergence.

    Args:
        N: Number of iterations to analyze
        L_value: Lipschitz constant value
        R_value: Initial distance bound value
        save_plot: Whether to save convergence plot
        output_dir: Directory to save outputs

    Returns:
        List of optimal values for each iteration
    """
    print(f"Analyzing numerical convergence for {N} iterations...")

    # Set up parameters
    R = pf.Parameter("R")

    # Create context and problem
    ctx_plt = make_ctx_gd(ctx_name="ctx_plt", N=N, stepsize=1/L)
    pb_plt = pf.PEPBuilder(ctx_plt)

    # Add initial condition: ||x_0 - x_*||^2 <= R
    pb_plt.add_initial_constraint(
        ((ctx_plt["x_0"] - ctx_plt["x_star"]) ** 2).le(R, name="initial_condition")
    )

    # Solve for each iteration
    opt_values = []
    for k in range(1, N):
        x_k = ctx_plt[f"x_{k}"]
        pb_plt.set_performance_metric(f(x_k) - f(ctx_plt["x_star"]))
        result = pb_plt.solve(resolve_parameters={"L": L_value, "R": R_value})
        opt_values.append(result.opt_value)
        print(f"Iteration {k}: Optimal value = {result.opt_value:.6f}")

    # Create convergence plot
    iters = np.arange(1, N)
    cont_iters = np.arange(1, N, 0.01)
    analytical_bound = L_value / (4 * cont_iters + 2)

    plt.figure(figsize=(10, 6))
    plt.plot(cont_iters, analytical_bound, 'r-',
             label=r"Analytical bound $\frac{L}{4k + 2}$", linewidth=2)
    plt.scatter(iters, opt_values, color="blue", marker="o", s=50,
               label="Numerical values", zorder=5)
    plt.xlabel("Iteration k")
    plt.ylabel("f(x_k) - f(x*)")
    plt.title("Gradient Descent Convergence Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    if save_plot:
        output_path = Path(output_dir) / "gradient_descent_convergence.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to: {output_path}")

    if not save_plot:
        plt.close()  # Don't show plot if skip-plot is enabled
    else:
        plt.show()

    return opt_values


def verify_analytical_proof(N: int = 2, L_value: float = 1.0, R_value: float = 1.0) -> dict:
    """
    Verify the analytical convergence proof for gradient descent.

    Args:
        N: Number of iterations for proof
        L_value: Lipschitz constant value
        R_value: Initial distance bound value

    Returns:
        Dictionary containing proof verification results
    """
    print(f"Verifying analytical proof for N={N} iterations...")

    # Set up symbolic computation
    R = pf.Parameter("R")

    # Create context for proof
    ctx_prf = make_ctx_gd(ctx_name="ctx_prf", N=sp.S(N), stepsize=1/L)
    pb_prf = pf.PEPBuilder(ctx_prf)
    pb_prf.add_initial_constraint(
        ((ctx_prf["x_0"] - ctx_prf["x_star"]) ** 2).le(R, name="initial_condition")
    )
    pb_prf.set_performance_metric(f(ctx_prf[f"x_{N}"]) - f(ctx_prf["x_star"]))

    # Solve the PEP problem
    result = pb_prf.solve(resolve_parameters={"L": L_value, "R": R_value})
    print(f"Optimal value: {result.opt_value:.10f}")

    # Get dual variables
    lamb_dense = result.get_scalar_constraint_dual_value_in_numpy(f)

    # Function to map tag to index
    def tag_to_index(tag, N=N):
        if (idx := tag.split("_")[1]).isdigit():
            return int(idx)
        elif idx == "star":
            return N + 1

    # Find relaxed constraints (keep only consecutive ones)
    relaxed_constraints = []
    for tag_i in lamb_dense.row_names:
        i = tag_to_index(tag_i)
        if i == N + 1:
            continue
        for tag_j in lamb_dense.col_names:
            j = tag_to_index(tag_j)
            if i < N and i + 1 == j:
                continue
            relaxed_constraints.append(f"f:{tag_i},{tag_j}")

    pb_prf.set_relaxed_constraints(relaxed_constraints)
    result = pb_prf.solve(resolve_parameters={"L": L_value, "R": R_value})

    # Extract dual variables
    tau_sol = result.dual_var_manager.dual_value("initial_condition")
    lamb_sol = result.get_scalar_constraint_dual_value_in_numpy(f)
    S_sol = result.get_gram_dual_matrix()

    # Verify closed-form lambda expression
    def lamb_analytical(tag_i, tag_j, N=N):
        i = tag_to_index(tag_i)
        j = tag_to_index(tag_j)
        if i == N + 1:  # x_star constraints
            if j == 0:
                return lamb_analytical("x_0", "x_1")
            elif j < N:
                return lamb_analytical(f"x_{j}", f"x_{j + 1}") - lamb_analytical(f"x_{j - 1}", f"x_{j}")
            elif j == N:
                return 1 - lamb_analytical(f"x_{N - 1}", f"x_{N}")
        if i < N and i + 1 == j:  # Consecutive constraints
            return j / (2 * N + 1 - j)
        return 0

    # Create analytical lambda matrix
    lamb_analytical_matrix = pf.pprint_labeled_matrix(
        lamb_analytical, lamb_sol.row_names, lamb_sol.col_names, return_matrix=True
    )

    # Check if analytical form matches numerical solution
    lambda_match = np.allclose(lamb_analytical_matrix, lamb_sol.matrix, atol=1e-4)
    print(f"Analytical lambda expression correct: {lambda_match}")

    # Verify S matrix expression using ExpressionManager
    pm = pf.ExpressionManager(ctx_prf, resolve_parameters={"L": L_value, "R": R_value})

    # Build analytical S expression
    x = ctx_prf.tracked_point(f)
    x_0 = ctx_prf["x_0"]
    x_star = ctx_prf["x_star"]

    tau_analytical = L_value / (4 * N + 2)
    grad_terms = sum(lamb_analytical(x_star.tag, x[i].tag) * f.grad(x[i]) for i in range(N + 1))
    z_N = x_star - x_0 + 1 / (2 * tau_analytical) * grad_terms
    iter_diff_square = tau_analytical * z_N**2

    # Additional gradient difference terms
    grad_diff_square = pf.Scalar.zero()
    for i in range(N + 1):
        for j in range(i + 1, N + 1):
            const_1 = (2 * N + 1) * lamb_analytical(x_star.tag, x[i].tag) - 1
            const_2 = lamb_analytical(x_star.tag, x[j].tag)
            grad_diff_square += (
                1 / (2 * L_value) * (const_1 * const_2 * (f.grad(x[i]) - f.grad(x[j])) ** 2)
            )

    S_analytical = iter_diff_square + grad_diff_square
    S_analytical_matrix = pm.eval_scalar(S_analytical).inner_prod_coords

    # Check if analytical S matches numerical solution
    S_match = np.allclose(S_analytical_matrix, S_sol.matrix, atol=1e-4)
    print(f"Analytical S expression correct: {S_match}")

    return {
        "optimal_value": result.opt_value,
        "lambda_correct": lambda_match,
        "S_correct": S_match,
        "tau_value": float(tau_sol),
        "analytical_bound": float(L_value / (4 * N + 2))
    }


def main():
    """Main function to run gradient descent analysis."""
    parser = argparse.ArgumentParser(description="Analyze Gradient Descent convergence using PEPFlow")
    parser.add_argument("--iterations", "-n", type=int, default=8,
                       help="Number of iterations for numerical analysis")
    parser.add_argument("--proof-steps", "-p", type=int, default=2,
                       help="Number of steps for analytical proof")
    parser.add_argument("--lipschitz", "-L", type=float, default=1.0,
                       help="Lipschitz constant value")
    parser.add_argument("--radius", "-R", type=float, default=1.0,
                       help="Initial distance bound")
    parser.add_argument("--output-dir", "-o", type=str, default="results/uc_001",
                       help="Output directory for plots and data")
    parser.add_argument("--skip-plot", action="store_true",
                       help="Skip plotting convergence analysis")

    args = parser.parse_args()

    print("=" * 60)
    print("GRADIENT DESCENT CONVERGENCE ANALYSIS")
    print("=" * 60)
    print(f"Parameters: L={args.lipschitz}, R={args.radius}")
    print()

    # Run numerical analysis
    try:
        opt_values = analyze_numerical_convergence(
            N=args.iterations,
            L_value=args.lipschitz,
            R_value=args.radius,
            save_plot=not args.skip_plot,
            output_dir=args.output_dir
        )
        print(f"Numerical analysis completed for {len(opt_values)} iterations.")
        print()

        # Run analytical verification
        proof_results = verify_analytical_proof(
            N=args.proof_steps,
            L_value=args.lipschitz,
            R_value=args.radius
        )

        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Optimal function value after {args.proof_steps} steps: {proof_results['optimal_value']:.10f}")
        print(f"Analytical bound L/(4N+2): {proof_results['analytical_bound']:.10f}")
        print(f"Lambda expression verified: {proof_results['lambda_correct']}")
        print(f"S matrix expression verified: {proof_results['S_correct']}")
        print()

        if proof_results['optimal_value'] <= proof_results['analytical_bound'] + 1e-8:
            print("✓ Convergence bound verified!")
        else:
            print("✗ Convergence bound verification failed.")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())