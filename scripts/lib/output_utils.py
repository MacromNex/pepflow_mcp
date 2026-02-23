"""
Output and visualization utilities for PEPFlow MCP scripts.

Functions for creating plots, saving results, and formatting output
across different optimization analysis scripts.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import sys

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
except ImportError as e:
    print(f"Warning: Plotting libraries not available: {e}")
    print("Install with: pip install matplotlib numpy")


def setup_matplotlib_style():
    """Set up consistent matplotlib style for all plots."""
    plt.style.use('default')
    mpl.rcParams['figure.figsize'] = (10, 6)
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.alpha'] = 0.3
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['legend.fontsize'] = 11


def create_convergence_plot(
    iterations: List[int],
    numerical_values: List[float],
    theoretical_values: List[float],
    algorithm_name: str,
    convergence_rate: str,
    output_path: Union[str, Path],
    log_scale: bool = True,
    theoretical_label: str = "Theoretical bound"
) -> None:
    """
    Create and save a convergence analysis plot.

    Args:
        iterations: List of iteration numbers
        numerical_values: Numerical optimal values
        theoretical_values: Theoretical bound values
        algorithm_name: Name of the algorithm
        convergence_rate: Convergence rate (e.g., "O(1/k)")
        output_path: Path to save the plot
        log_scale: Whether to use logarithmic scale
        theoretical_label: Label for theoretical curve
    """
    setup_matplotlib_style()

    plt.figure(figsize=(10, 6))

    # Plot theoretical bound
    continuous_iters = np.linspace(iterations[0], iterations[-1], 1000)
    if "O(1/k^2)" in convergence_rate or "AGM" in algorithm_name.upper():
        # For AGM, use quadratic convergence
        L_approx = theoretical_values[0] * (2 * iterations[0]**2) if theoretical_values else 1.0
        continuous_bound = L_approx / (2 * continuous_iters**2)
    else:
        # For GD, use linear convergence
        L_approx = theoretical_values[0] * (4 * iterations[0] + 2) if theoretical_values else 1.0
        continuous_bound = L_approx / (4 * continuous_iters + 2)

    plt.plot(continuous_iters, continuous_bound, 'r-',
             label=f"{theoretical_label} ({convergence_rate})", linewidth=2)

    # Plot numerical values
    plt.scatter(iterations, numerical_values, color="blue", marker="o", s=50,
               label="Numerical values", zorder=5)

    # Plot theoretical points if available
    if theoretical_values:
        plt.scatter(iterations, theoretical_values, color="red", marker="x", s=60,
                   label="Theoretical points", zorder=4)

    plt.xlabel("Iteration k")
    plt.ylabel("f(x_k) - f(x*)")
    plt.title(f"{algorithm_name} - {convergence_rate} Convergence Analysis")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if log_scale:
        plt.yscale('log')

    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Convergence plot saved to: {output_path}")
    plt.close()


def create_theta_sequence_plot(
    iterations: List[int],
    theta_values: List[float],
    output_path: Union[str, Path]
) -> None:
    """
    Create and save a plot of the AGM theta sequence.

    Args:
        iterations: List of iteration numbers
        theta_values: Corresponding theta values
        output_path: Path to save the plot
    """
    setup_matplotlib_style()

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, theta_values, 'g-o', linewidth=2, markersize=6)
    plt.xlabel("Iteration k")
    plt.ylabel("θ_k")
    plt.title("Optimal Theta Sequence for Accelerated Gradient Method")
    plt.grid(True, alpha=0.3)

    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Theta sequence plot saved to: {output_path}")
    plt.close()


def create_algorithm_comparison_plot(
    comparison_results: Dict[str, Any],
    output_path: Union[str, Path],
    max_iterations: int = 10
) -> None:
    """
    Create and save a comparison plot for multiple algorithms.

    Args:
        comparison_results: Results from algorithm comparison
        output_path: Path to save the plot
        max_iterations: Maximum number of iterations to plot
    """
    setup_matplotlib_style()

    plt.figure(figsize=(12, 8))

    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']

    for i, (algorithm, result) in enumerate(comparison_results.items()):
        if result.get("status") == "failed":
            continue

        results = result.get("results", [])
        if not results:
            continue

        # Extract iterations and values
        iterations = []
        values = []
        for r in results[:max_iterations]:
            if r.get("status") == "success":
                iterations.append(r["iteration"])
                values.append(r["optimal_value"])

        if not iterations:
            continue

        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        plt.semilogy(iterations, values, f'{color}-{marker}',
                    label=f"{algorithm.replace('_', ' ').title()}",
                    linewidth=2, markersize=6)

    plt.xlabel("Iteration k")
    plt.ylabel("f(x_k) - f(x*) [log scale]")
    plt.title("Algorithm Comparison - Convergence Rates")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Algorithm comparison plot saved to: {output_path}")
    plt.close()


def save_results_json(
    results: Dict[str, Any],
    output_path: Union[str, Path],
    indent: int = 2
) -> None:
    """
    Save results to JSON file with proper formatting.

    Args:
        results: Results dictionary
        output_path: Path to save JSON file
        indent: JSON indentation level
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    serializable_results = convert_numpy_types(results)

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=indent)

    print(f"Results saved to: {output_path}")


def save_convergence_data_csv(
    iterations: List[int],
    numerical_values: List[float],
    theoretical_values: Optional[List[float]],
    output_path: Union[str, Path]
) -> None:
    """
    Save convergence data to CSV file.

    Args:
        iterations: List of iteration numbers
        numerical_values: Numerical optimal values
        theoretical_values: Theoretical bound values (optional)
        output_path: Path to save CSV file
    """
    import csv

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        if theoretical_values:
            writer.writerow(['Iteration', 'Numerical_Value', 'Theoretical_Value'])
            # Write data
            for i, (iter_num, num_val) in enumerate(zip(iterations, numerical_values)):
                theo_val = theoretical_values[i] if i < len(theoretical_values) else ''
                writer.writerow([iter_num, num_val, theo_val])
        else:
            writer.writerow(['Iteration', 'Numerical_Value'])
            # Write data
            for iter_num, num_val in zip(iterations, numerical_values):
                writer.writerow([iter_num, num_val])

    print(f"Convergence data saved to: {output_path}")


def create_execution_summary(
    algorithm: str,
    config: Dict[str, Any],
    results: Dict[str, Any],
    execution_time: Optional[float] = None
) -> str:
    """
    Create a formatted execution summary.

    Args:
        algorithm: Algorithm name
        config: Configuration used
        results: Execution results
        execution_time: Execution time in seconds (optional)

    Returns:
        Formatted summary string
    """
    summary_lines = [
        "=" * 60,
        f"EXECUTION SUMMARY: {algorithm.upper()}",
        "=" * 60,
        ""
    ]

    # Configuration summary
    params = config.get("parameters", {})
    analysis = config.get("analysis", {})

    summary_lines.extend([
        "Configuration:",
        f"  Algorithm: {algorithm}",
        f"  Iterations: {analysis.get('iterations', 'N/A')}",
        f"  Lipschitz constant: {params.get('lipschitz_constant', 'N/A')}",
        f"  Initial radius: {params.get('initial_radius', 'N/A')}",
        ""
    ])

    # Results summary
    if "analytical_results" in results:
        ar = results["analytical_results"]
        summary_lines.extend([
            "Analytical Results:",
            f"  Optimal value: {ar.get('optimal_value', 'N/A'):.6f}",
            f"  Theoretical bound: {ar.get('analytical_bound', 'N/A'):.6f}",
            f"  Convergence verified: {ar.get('lambda_correct', 'N/A')}",
            ""
        ])

    if "numerical_results" in results:
        nr = results["numerical_results"]
        summary_lines.extend([
            "Numerical Results:",
            f"  Iterations computed: {len(nr)}",
            f"  Best value: {min(nr) if nr else 'N/A':.6f}",
            ""
        ])

    # Performance summary
    if execution_time:
        summary_lines.extend([
            f"Execution time: {execution_time:.2f} seconds",
            ""
        ])

    summary_lines.append("=" * 60)

    return "\n".join(summary_lines)


def ensure_output_directory(output_path: Union[str, Path]) -> Path:
    """
    Ensure output directory exists and return Path object.

    Args:
        output_path: Output directory path

    Returns:
        Path object for the directory
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_execution_log(
    log_content: str,
    output_dir: Union[str, Path],
    filename: str = "execution.log"
) -> None:
    """
    Save execution log to file.

    Args:
        log_content: Content to save
        output_dir: Output directory
        filename: Log filename
    """
    output_dir = ensure_output_directory(output_dir)
    log_path = output_dir / filename

    with open(log_path, 'w') as f:
        f.write(log_content)

    print(f"Execution log saved to: {log_path}")


def format_table_results(
    results: List[Dict[str, Any]],
    columns: List[str],
    headers: List[str]
) -> str:
    """
    Format results as a table string.

    Args:
        results: List of result dictionaries
        columns: Column keys to extract
        headers: Column headers

    Returns:
        Formatted table string
    """
    if not results:
        return "No results to display"

    # Calculate column widths
    col_widths = [len(header) for header in headers]
    for result in results:
        for i, col in enumerate(columns):
            value_str = str(result.get(col, 'N/A'))
            col_widths[i] = max(col_widths[i], len(value_str))

    # Create separator
    separator = "+" + "+".join(["-" * (width + 2) for width in col_widths]) + "+"

    # Create table
    table_lines = [separator]

    # Header row
    header_row = "|" + "|".join([f" {header:^{col_widths[i]}} " for i, header in enumerate(headers)]) + "|"
    table_lines.extend([header_row, separator])

    # Data rows
    for result in results:
        values = [str(result.get(col, 'N/A')) for col in columns]
        data_row = "|" + "|".join([f" {value:^{col_widths[i]}} " for i, value in enumerate(values)]) + "|"
        table_lines.append(data_row)

    table_lines.append(separator)

    return "\n".join(table_lines)