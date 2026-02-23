"""
PEPFlow utility functions for optimization analysis.

Common functions for creating contexts, solving problems, and handling
PEPFlow-specific operations across different optimization algorithms.
"""

from typing import Dict, Any, Optional, List
import sys

try:
    import pepflow as pf
    import sympy as sp
    import numpy as np
except ImportError as e:
    print(f"Error: Required package not found: {e}")
    print("Please install PEPFlow framework: mamba run -p ./env pip install -e repo/PEPFlow")
    sys.exit(1)

# ==============================================================================
# Global PEPFlow Parameters (shared across scripts)
# ==============================================================================

def get_global_pepflow_parameters():
    """Get the global L and f parameters used by all algorithms."""
    L = pf.Parameter("L")
    f = pf.SmoothConvexFunction(is_basis=True, tags=["f"], L=L)
    return L, f

# ==============================================================================
# Common Context Setup Functions
# ==============================================================================

def add_initial_constraints(
    pb: pf.PEPBuilder,
    ctx: pf.PEPContext,
    R: pf.Parameter,
    include_prev_point: bool = False
) -> None:
    """
    Add standard initial constraints to a PEP problem.

    Args:
        pb: PEPBuilder instance
        ctx: PEPContext instance
        R: Radius parameter
        include_prev_point: Whether to include constraint for x_{-1} (for momentum methods)
    """
    # Standard initial condition: ||x_0 - x_*||^2 <= R
    pb.add_initial_constraint(
        ((ctx["x_0"] - ctx["x_star"]) ** 2).le(R, name="initial_condition_x0")
    )

    # For momentum methods, also constrain x_{-1}
    if include_prev_point:
        pb.add_initial_constraint(
            ((ctx["x_-1"] - ctx["x_star"]) ** 2).le(R, name="initial_condition_x_minus1")
        )

def solve_pep_iterations(
    ctx: pf.PEPContext,
    pb: pf.PEPBuilder,
    f: pf.SmoothConvexFunction,
    N: int,
    L_value: float,
    R_value: float,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Solve PEP problem for multiple iterations.

    Args:
        ctx: PEPContext instance
        pb: PEPBuilder instance
        f: Smooth convex function
        N: Number of iterations to solve
        L_value: Lipschitz constant value
        R_value: Initial radius value
        verbose: Whether to print iteration results

    Returns:
        List of results for each iteration
    """
    results = []

    for k in range(1, N):
        x_k = ctx[f"x_{k}"]
        pb.set_performance_metric(f(x_k) - f(ctx["x_star"]))

        try:
            result = pb.solve(resolve_parameters={"L": L_value, "R": R_value})
            results.append({
                "iteration": k,
                "optimal_value": result.opt_value,
                "status": "success",
                "result_object": result
            })
            if verbose:
                print(f"Iteration {k}: Optimal value = {result.opt_value:.6f}")
        except Exception as e:
            results.append({
                "iteration": k,
                "optimal_value": None,
                "status": "failed",
                "error": str(e)
            })
            if verbose:
                print(f"Iteration {k}: Failed with error: {e}")

    return results

# ==============================================================================
# Dual Variable Analysis
# ==============================================================================

def extract_dual_variables(result, f: pf.SmoothConvexFunction) -> Dict[str, Any]:
    """
    Extract and process dual variables from PEP solution.

    Args:
        result: PEP solution result
        f: Smooth convex function

    Returns:
        Dictionary containing dual variable information
    """
    try:
        lamb_dense = result.get_scalar_constraint_dual_value_in_numpy(f)
        dual_info = {
            "has_dual_vars": True,
            "dual_matrix_shape": lamb_dense.shape if hasattr(lamb_dense, 'shape') else None,
            "dual_representation": str(lamb_dense) if lamb_dense is not None else None
        }
    except Exception as e:
        dual_info = {
            "has_dual_vars": False,
            "error": str(e),
            "dual_representation": None
        }

    return dual_info

# ==============================================================================
# Convergence Analysis
# ==============================================================================

def verify_convergence_bound(
    optimal_value: float,
    theoretical_bound: float,
    tolerance: float = 1e-8
) -> Dict[str, Any]:
    """
    Verify that optimal value satisfies theoretical convergence bound.

    Args:
        optimal_value: Computed optimal value
        theoretical_bound: Theoretical convergence bound
        tolerance: Tolerance for numerical comparison

    Returns:
        Dictionary with verification results
    """
    difference = optimal_value - theoretical_bound
    is_satisfied = optimal_value <= theoretical_bound + tolerance

    return {
        "optimal_value": optimal_value,
        "theoretical_bound": theoretical_bound,
        "difference": difference,
        "bound_satisfied": is_satisfied,
        "tolerance_used": tolerance,
        "relative_error": abs(difference / theoretical_bound) if theoretical_bound != 0 else float('inf')
    }

def compute_gradient_descent_bound(L_value: float, N: int) -> float:
    """Compute theoretical bound for gradient descent: L/(4*N+2)"""
    return L_value / (4 * N + 2)

def compute_agm_bound(L_value: float, theta_N: float) -> float:
    """Compute theoretical bound for AGM: L/(2*theta_N^2)"""
    return L_value / (2 * theta_N**2)

# ==============================================================================
# Algorithm-Specific Utilities
# ==============================================================================

def create_standard_gradient_context(
    ctx_name: str,
    N: int,
    L: pf.Parameter,
    f: pf.SmoothConvexFunction
) -> pf.PEPContext:
    """
    Create standard gradient descent context.

    Args:
        ctx_name: Context name
        N: Number of iterations
        L: Lipschitz parameter
        f: Smooth convex function

    Returns:
        Configured PEPContext
    """
    ctx = pf.PEPContext(ctx_name).set_as_current()
    x = pf.Vector(is_basis=True, tags=["x_0"])
    f.set_stationary_point("x_star")

    for i in range(N):
        x = x - 1/L * f.grad(x)
        x.add_tag(f"x_{i + 1}")

    return ctx

def create_momentum_context(
    ctx_name: str,
    N: int,
    L: pf.Parameter,
    f: pf.SmoothConvexFunction,
    momentum: float
) -> pf.PEPContext:
    """
    Create momentum-based optimization context (heavy ball).

    Args:
        ctx_name: Context name
        N: Number of iterations
        L: Lipschitz parameter
        f: Smooth convex function
        momentum: Momentum parameter

    Returns:
        Configured PEPContext
    """
    ctx = pf.PEPContext(ctx_name).set_as_current()
    x = pf.Vector(is_basis=True, tags=["x_0"])
    x_prev = pf.Vector(is_basis=True, tags=["x_-1"])
    f.set_stationary_point("x_star")

    for i in range(N):
        # Heavy ball update: x_{k+1} = x_k - (1/L)*∇f(x_k) + β*(x_k - x_{k-1})
        x_new = x - 1/L * f.grad(x) + momentum * (x - x_prev)
        x_prev = x
        x = x_new
        x.add_tag(f"x_{i + 1}")

    return ctx

# ==============================================================================
# Performance Metrics
# ==============================================================================

def find_best_iteration(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Find the iteration with best (minimum) optimal value.

    Args:
        results: List of iteration results

    Returns:
        Best result dictionary or None if no successful iterations
    """
    successful_results = [r for r in results if r["status"] == "success"]
    if not successful_results:
        return None

    best_result = min(successful_results, key=lambda x: x["optimal_value"])
    return {
        "best_iteration": best_result["iteration"],
        "best_optimal_value": best_result["optimal_value"],
        "performance_point": f"x_{best_result['iteration']}",
        "total_successful": len(successful_results),
        "total_attempted": len(results)
    }

# ==============================================================================
# Error Handling
# ==============================================================================

def validate_pepflow_environment():
    """
    Validate that PEPFlow environment is properly set up.

    Raises:
        ImportError: If required packages are not available
        RuntimeError: If PEPFlow is not properly configured
    """
    try:
        # Test basic PEPFlow functionality
        L_test = pf.Parameter("L_test")
        f_test = pf.SmoothConvexFunction(is_basis=True, tags=["f_test"], L=L_test)
        ctx_test = pf.PEPContext("test_ctx")

        # Clean up
        del L_test, f_test, ctx_test

    except Exception as e:
        raise RuntimeError(f"PEPFlow environment validation failed: {e}")

# ==============================================================================
# Utility Functions
# ==============================================================================

def format_results_summary(results: List[Dict[str, Any]], algorithm: str) -> str:
    """
    Format results into a readable summary string.

    Args:
        results: List of iteration results
        algorithm: Algorithm name

    Returns:
        Formatted summary string
    """
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]

    summary = [
        f"Algorithm: {algorithm}",
        f"Total iterations: {len(results)}",
        f"Successful: {len(successful)}",
        f"Failed: {len(failed)}"
    ]

    if successful:
        best = min(successful, key=lambda x: x["optimal_value"])
        summary.extend([
            f"Best optimal value: {best['optimal_value']:.6f}",
            f"Best iteration: {best['iteration']}"
        ])

    return "\n".join(summary)