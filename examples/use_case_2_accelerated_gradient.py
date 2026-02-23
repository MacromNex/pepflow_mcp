#!/usr/bin/env python3
"""
Use Case 2: Accelerated Gradient Method (AGM) Analysis

This script demonstrates Nesterov's accelerated gradient method analysis using PEPFlow.
AGM achieves O(1/k²) convergence rate for L-smooth convex functions, which is optimal.

Based on "A Method of Solving a Convex Programming Problem with Convergence Rate O(1/k^2)"
by Yurii Nesterov (1983) and analysis from "Optimized First-Order Methods for Smooth
Convex Minimization" by Donghwan Kim and Jeffrey A. Fessler (2016).

Author: Extracted from PEPFlow examples
License: Apache 2.0
"""

import argparse
import functools
import pepflow as pf
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import itertools
from pathlib import Path

# Define global parameters and function (following the notebook pattern)
L = pf.Parameter("L")
f = pf.SmoothConvexFunction(is_basis=True, tags=["f"], L=L)


@functools.cache
def theta(i):
    """
    Compute the theta sequence for AGM.
    theta_0 = 0, theta_k = (1 + sqrt(1 + 4*theta_{k-1}^2)) / 2

    Args:
        i: Iteration index

    Returns:
        theta_i value
    """
    if i == -1:
        return 0
    return 1 / sp.S(2) * (sp.S(1) + sp.sqrt(4 * theta(i - 1) ** 2 + sp.S(1)))


def make_ctx_agm(ctx_name: str, N: int, stepsize) -> pf.PEPContext:
    """
    Create PEPContext for Accelerated Gradient Method analysis.
    Based on the original notebook implementation.

    Args:
        ctx_name: Name for the context
        N: Number of iterations
        stepsize: Step size parameter (typically 1/L)

    Returns:
        PEPContext configured for AGM
    """
    ctx_agm = pf.PEPContext(ctx_name).set_as_current()
    x = pf.Vector(is_basis=True, tags=["x_0"])
    f.set_stationary_point("x_star")
    z = x
    for i in range(N):
        y = x - stepsize * f.grad(x)
        z = z - stepsize * theta(i) * f.grad(x)
        x = (1 - 1 / theta(i + 1)) * y + 1 / theta(i + 1) * z
        z.add_tag(f"z_{i + 1}")
        x.add_tag(f"x_{i + 1}")
    return ctx_agm


def analyze_agm_convergence(N: int = 5, L_value: float = 1.0, R_value: float = 1.0,
                           save_plot: bool = True, output_dir: str = "examples/data") -> list:
    """
    Perform numerical analysis of AGM convergence.

    Args:
        N: Number of iterations to analyze
        L_value: Lipschitz constant value
        R_value: Initial distance bound value
        save_plot: Whether to save convergence plot
        output_dir: Directory to save outputs

    Returns:
        List of optimal values for each iteration
    """
    print(f"Analyzing AGM convergence for {N} iterations...")

    # Set up parameters
    R = pf.Parameter("R")

    # Solve for each iteration
    opt_values = []
    for k in range(1, N):
        ctx_plt = make_ctx_agm(ctx_name=f"ctx_plt_{k}", N=k, stepsize=1/L)
        pb_plt = pf.PEPBuilder(ctx_plt)
        pb_plt.add_initial_constraint(
            ((ctx_plt["x_0"] - ctx_plt["x_star"]) ** 2).le(R, name="initial_condition")
        )
        x_k = ctx_plt[f"x_{k}"]
        pb_plt.set_performance_metric(f(x_k) - f(ctx_plt["x_star"]))
        result = pb_plt.solve(resolve_parameters={"L": L_value, "R": R_value})
        opt_values.append(result.opt_value)
        print(f"Iteration {k}: Optimal value = {result.opt_value:.6f}")

    # Create convergence plot
    iters = np.arange(1, N)
    analytical_values = [L_value / (2 * theta(i) ** 2) for i in iters]

    plt.figure(figsize=(10, 6))
    plt.scatter(iters, analytical_values, color="red", marker="x", s=100,
               label=r"Analytical bound $\frac{L}{2\theta_N^2}$")
    plt.scatter(iters, opt_values, color="blue", marker="o", s=50,
               label="Numerical values", zorder=5)
    plt.xlabel("Iteration k")
    plt.ylabel("f(x_k) - f(x*)")
    plt.title("Accelerated Gradient Method Convergence Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    if save_plot:
        output_path = Path(output_dir) / "agm_convergence.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to: {output_path}")

    if not save_plot:
        plt.close()  # Don't show plot if skip-plot is enabled
    else:
        plt.show()

    return opt_values


def verify_agm_proof(N: int = 3, L_value: float = 1.0, R_value: float = 1.0) -> dict:
    """
    Verify the analytical convergence proof for AGM.

    Args:
        N: Number of iterations for proof
        L_value: Lipschitz constant value
        R_value: Initial distance bound value

    Returns:
        Dictionary containing proof verification results
    """
    print(f"Verifying AGM proof for N={N} iterations...")

    # Set up symbolic computation
    R = pf.Parameter("R")

    # Create context for proof
    ctx_prf = make_ctx_agm(ctx_name="ctx_prf", N=sp.S(N), stepsize=1/L)
    pb_prf = pf.PEPBuilder(ctx_prf)
    pb_prf.add_initial_constraint(
        ((ctx_prf["x_0"] - ctx_prf["x_star"]) ** 2).le(R, name="initial_condition")
    )
    pb_prf.set_performance_metric(f(ctx_prf[f"x_{N}"]) - f(ctx_prf["x_star"]))

    # Solve the PEP problem
    result = pb_prf.solve(resolve_parameters={"L": L_value, "R": R_value})
    print(f"Optimal value: {result.opt_value:.10f}")

    # Compare with theoretical bound
    desired_upper_bound = float(L_value / (2 * theta(N) ** 2))
    print(f"Desired upper bound: {desired_upper_bound:.10f}")

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

    # Add dual constraint to match analytical bound
    pb_prf.add_dual_val_constraint("initial_condition", "==", desired_upper_bound)

    # Set lambda constraints according to paper
    for i in range(N + 1):
        pb_prf.add_dual_val_constraint(
            f"f:x_{i},x_{i + 1}", "==", theta(i) ** 2 / theta(N) ** 2
        )

    # Solve dual problem
    result_dual = pb_prf.solve_dual(resolve_parameters={"L": L_value, "R": R_value})
    print(f"Dual optimal value: {result_dual.opt_value:.10f}")

    # Extract dual variables
    tau_sol = result_dual.dual_var_manager.dual_value("initial_condition")
    lamb_sol = result_dual.get_scalar_constraint_dual_value_in_numpy(f)
    S_sol = result_dual.get_gram_dual_matrix()

    # Verify closed-form lambda expression
    def lamb_analytical(tag_i, tag_j, N=N):
        i = tag_to_index(tag_i)
        j = tag_to_index(tag_j)
        if i == N + 1:  # x_star constraints
            if j < N + 1:
                return theta(j) / theta(N) ** 2
        if i < N and i + 1 == j:  # consecutive constraints
            return theta(i) ** 2 / theta(N) ** 2
        return 0

    # Create analytical lambda matrix
    lamb_analytical_matrix = pf.pprint_labeled_matrix(
        lamb_analytical, lamb_sol.row_names, lamb_sol.col_names, return_matrix=True
    )

    # Check if analytical form matches numerical solution
    lambda_match = np.allclose(lamb_analytical_matrix, lamb_sol.matrix, atol=1e-3)
    print(f"Analytical lambda expression correct: {lambda_match}")

    # Verify S matrix expression using ExpressionManager
    pm = pf.ExpressionManager(ctx_prf, resolve_parameters={"L": L_value, "R": R_value})

    # Build analytical S expression (simplified version)
    z_N = ctx_prf[f"z_{N}"]
    x_N = ctx_prf[f"x_{N}"]
    x_star = ctx_prf["x_star"]

    S_guess1 = L_value / theta(N) ** 2 * 1 / 2 * (z_N - theta(N) / L_value * f.grad(x_N) - x_star) ** 2

    coef = 1 / 2 * 1 / theta(N) ** 2 / L_value
    S_guess2 = (
        coef * sum(theta(i) ** 2 * f.grad(ctx_prf[f"x_{i}"]) ** 2 for i in range(N))
    )

    S_analytical = S_guess1 + S_guess2
    S_analytical_matrix = pm.eval_scalar(S_analytical).inner_prod_coords

    # Check if analytical S matches numerical solution
    S_match = np.allclose(S_analytical_matrix, S_sol.matrix, atol=1e-3)
    print(f"Analytical S expression correct: {S_match}")

    return {
        "optimal_value": result.opt_value,
        "dual_optimal_value": result_dual.opt_value,
        "lambda_correct": lambda_match,
        "S_correct": S_match,
        "tau_value": float(tau_sol),
        "analytical_bound": desired_upper_bound,
        "bound_satisfied": result.opt_value <= desired_upper_bound
    }


def main():
    """Main function to run AGM analysis."""
    parser = argparse.ArgumentParser(description="Analyze Accelerated Gradient Method using PEPFlow")
    parser.add_argument("--iterations", "-n", type=int, default=5,
                       help="Number of iterations for numerical analysis")
    parser.add_argument("--proof-steps", "-p", type=int, default=3,
                       help="Number of steps for analytical proof")
    parser.add_argument("--lipschitz", "-L", type=float, default=1.0,
                       help="Lipschitz constant value")
    parser.add_argument("--radius", "-R", type=float, default=1.0,
                       help="Initial distance bound")
    parser.add_argument("--output-dir", "-o", type=str, default="results/uc_002",
                       help="Output directory for plots and data")
    parser.add_argument("--skip-plot", action="store_true",
                       help="Skip plotting convergence analysis")

    args = parser.parse_args()

    print("=" * 60)
    print("ACCELERATED GRADIENT METHOD ANALYSIS")
    print("=" * 60)
    print(f"Parameters: L={args.lipschitz}, R={args.radius}")
    print()

    # Run numerical analysis
    try:
        opt_values = analyze_agm_convergence(
            N=args.iterations,
            L_value=args.lipschitz,
            R_value=args.radius,
            save_plot=not args.skip_plot,
            output_dir=args.output_dir
        )
        print(f"Numerical analysis completed for {len(opt_values)} iterations.")
        print()

        # Run analytical verification
        proof_results = verify_agm_proof(
            N=args.proof_steps,
            L_value=args.lipschitz,
            R_value=args.radius
        )

        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Optimal function value after {args.proof_steps} steps: {proof_results['optimal_value']:.10f}")
        print(f"Dual optimal value: {proof_results['dual_optimal_value']:.10f}")
        print(f"Analytical bound L/(2*theta_N^2): {proof_results['analytical_bound']:.10f}")
        print(f"Lambda expression verified: {proof_results['lambda_correct']}")
        print(f"S matrix expression verified: {proof_results['S_correct']}")
        print(f"Convergence bound satisfied: {proof_results['bound_satisfied']}")
        print()

        if proof_results['bound_satisfied']:
            print("✓ AGM convergence bound verified!")
        else:
            print("✗ AGM convergence bound verification failed.")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())