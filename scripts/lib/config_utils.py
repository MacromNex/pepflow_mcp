"""
Configuration management utilities for PEPFlow MCP scripts.

Functions for loading, validating, and merging configuration files
across different optimization analysis scripts.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union


def load_config_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in config file {config_path}: {e}")


def merge_configs(
    default_config: Dict[str, Any],
    file_config: Optional[Dict[str, Any]] = None,
    user_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Merge configurations with proper precedence.

    Precedence (lowest to highest):
    1. default_config
    2. file_config
    3. user_config
    4. kwargs

    Args:
        default_config: Default configuration dictionary
        file_config: Configuration from file (optional)
        user_config: User-provided configuration (optional)
        **kwargs: Direct parameter overrides

    Returns:
        Merged configuration dictionary
    """
    final_config = default_config.copy()

    # Deep merge nested dictionaries
    def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    if file_config:
        final_config = deep_merge(final_config, file_config)

    if user_config:
        final_config = deep_merge(final_config, user_config)

    if kwargs:
        final_config = deep_merge(final_config, kwargs)

    return final_config


def validate_config(config: Dict[str, Any], required_fields: Dict[str, type]) -> None:
    """
    Validate configuration contains required fields with correct types.

    Args:
        config: Configuration dictionary to validate
        required_fields: Dict mapping field names to expected types

    Raises:
        ValueError: If required field is missing or has wrong type
    """
    def check_nested_field(data: Dict[str, Any], field_path: str, expected_type: type):
        keys = field_path.split('.')
        current = data

        for key in keys[:-1]:
            if key not in current:
                raise ValueError(f"Missing configuration section: {key}")
            current = current[key]

        final_key = keys[-1]
        if final_key not in current:
            raise ValueError(f"Missing required configuration field: {field_path}")

        value = current[final_key]
        if not isinstance(value, expected_type):
            raise ValueError(
                f"Configuration field {field_path} should be {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )

    for field_path, expected_type in required_fields.items():
        check_nested_field(config, field_path, expected_type)


def get_algorithm_config(config: Dict[str, Any], algorithm: str) -> Dict[str, Any]:
    """
    Extract algorithm-specific configuration.

    Args:
        config: Full configuration dictionary
        algorithm: Algorithm name

    Returns:
        Algorithm-specific configuration

    Raises:
        ValueError: If algorithm not supported or configuration missing
    """
    if "algorithm_settings" not in config:
        raise ValueError("No algorithm_settings section in configuration")

    if algorithm not in config["algorithm_settings"]:
        available = list(config["algorithm_settings"].keys())
        raise ValueError(f"Algorithm {algorithm} not supported. Available: {available}")

    return config["algorithm_settings"][algorithm]


def create_default_gradient_descent_config() -> Dict[str, Any]:
    """Create default configuration for gradient descent analysis."""
    return {
        "algorithm": {
            "name": "gradient_descent",
            "stepsize": "1/L"
        },
        "analysis": {
            "iterations": 8,
            "proof_steps": 2,
            "convergence_rate": "O(1/k)"
        },
        "parameters": {
            "lipschitz_constant": 1.0,
            "initial_radius": 1.0,
            "tolerance": 1e-8
        },
        "output": {
            "save_plot": True,
            "plot_format": "png",
            "include_verification": True,
            "output_dir": "results/gradient_descent"
        },
        "numerical_settings": {
            "solver": "OSQP",
            "precision": 1e-10,
            "log_scale_plot": True
        }
    }


def create_default_agm_config() -> Dict[str, Any]:
    """Create default configuration for accelerated gradient method."""
    return {
        "algorithm": {
            "name": "accelerated_gradient_method",
            "type": "Nesterov",
            "stepsize": "1/L",
            "momentum": "adaptive"
        },
        "analysis": {
            "iterations": 5,
            "proof_steps": 3,
            "convergence_rate": "O(1/k^2)"
        },
        "parameters": {
            "lipschitz_constant": 1.0,
            "initial_radius": 1.0,
            "tolerance": 1e-8
        },
        "output": {
            "save_plot": True,
            "plot_format": "png",
            "include_verification": True,
            "include_theta_sequence": True,
            "output_dir": "results/accelerated_gradient"
        },
        "numerical_settings": {
            "solver": "OSQP",
            "precision": 1e-10,
            "log_scale_plot": True,
            "dual_analysis": True
        }
    }


def create_default_pep_config() -> Dict[str, Any]:
    """Create default configuration for general PEP framework."""
    return {
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


def save_config_to_file(config: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save configuration dictionary to JSON file.

    Args:
        config: Configuration dictionary
        output_path: Path to save configuration file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)


def extract_cli_overrides(args) -> Dict[str, Any]:
    """
    Extract configuration overrides from CLI arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Dictionary of configuration overrides
    """
    overrides = {}

    # Common parameter mappings
    if hasattr(args, 'iterations') and args.iterations is not None:
        overrides.setdefault("analysis", {})["iterations"] = args.iterations

    if hasattr(args, 'proof_steps') and args.proof_steps is not None:
        overrides.setdefault("analysis", {})["proof_steps"] = args.proof_steps

    if hasattr(args, 'lipschitz') and args.lipschitz is not None:
        overrides.setdefault("parameters", {})["lipschitz_constant"] = args.lipschitz

    if hasattr(args, 'radius') and args.radius is not None:
        overrides.setdefault("parameters", {})["initial_radius"] = args.radius

    if hasattr(args, 'skip_plot') and args.skip_plot:
        overrides.setdefault("output", {})["save_plot"] = False

    if hasattr(args, 'algorithm') and args.algorithm is not None:
        overrides.setdefault("analysis", {})["algorithm"] = args.algorithm
        if args.algorithm == "comparison":
            overrides["analysis"]["comparison_mode"] = True

    if hasattr(args, 'momentum') and args.momentum is not None:
        overrides.setdefault("algorithm_settings", {}).setdefault("heavy_ball", {})["momentum"] = args.momentum

    return overrides


def get_required_fields_for_algorithm(algorithm: str) -> Dict[str, type]:
    """
    Get required configuration fields for specific algorithm.

    Args:
        algorithm: Algorithm name

    Returns:
        Dictionary mapping required field paths to expected types
    """
    common_fields = {
        "parameters.lipschitz_constant": float,
        "parameters.initial_radius": float,
        "output.output_dir": str,
        "numerical_settings.solver": str
    }

    algorithm_specific = {
        "gradient_descent": {
            "analysis.iterations": int,
            "analysis.proof_steps": int
        },
        "accelerated_gradient_method": {
            "analysis.iterations": int,
            "analysis.proof_steps": int
        },
        "pep_optimization": {
            "analysis.algorithm": str,
            "analysis.iterations": int
        }
    }

    required = common_fields.copy()
    if algorithm in algorithm_specific:
        required.update(algorithm_specific[algorithm])

    return required