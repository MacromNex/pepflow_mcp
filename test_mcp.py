#!/usr/bin/env python3
"""
Test script for MCP server functionality.
"""

import subprocess
import sys
import json
import time
from pathlib import Path

def run_mcp_command(tool_name, args=None):
    """Run an MCP tool command via subprocess."""
    cmd = [sys.executable, "src/server.py", "tools", "call", tool_name]
    if args:
        cmd.extend(["--arguments", json.dumps(args)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=Path(__file__).parent
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"

def test_utility_tools():
    """Test utility tools that don't require PEPFlow."""
    print("Testing utility tools...")

    # Test get_available_algorithms
    returncode, stdout, stderr = run_mcp_command("get_available_algorithms")
    print(f"get_available_algorithms: {returncode}")
    if returncode == 0:
        print("✓ Available algorithms tool works")
    else:
        print(f"✗ Failed: {stderr}")

    # Test get_default_configs
    returncode, stdout, stderr = run_mcp_command("get_default_configs")
    print(f"get_default_configs: {returncode}")
    if returncode == 0:
        print("✓ Default configs tool works")
    else:
        print(f"✗ Failed: {stderr}")

    # Test validate_pepflow_installation
    returncode, stdout, stderr = run_mcp_command("validate_pepflow_installation")
    print(f"validate_pepflow_installation: {returncode}")
    if returncode == 0:
        try:
            result = json.loads(stdout)
            if result.get("pepflow_available"):
                print("✓ PEPFlow is available")
                return True
            else:
                print("✗ PEPFlow not available")
                return False
        except:
            print("✗ Failed to parse validation result")
            return False
    else:
        print(f"✗ Validation failed: {stderr}")
        return False

def test_job_management():
    """Test job management tools."""
    print("\nTesting job management tools...")

    # Test list_jobs
    returncode, stdout, stderr = run_mcp_command("list_jobs")
    print(f"list_jobs: {returncode}")
    if returncode == 0:
        print("✓ Job listing works")
    else:
        print(f"✗ Failed: {stderr}")

def test_sync_tools():
    """Test synchronous tools (if PEPFlow is available)."""
    print("\nTesting synchronous analysis tools...")

    # Test gradient descent analysis with minimal parameters
    args = {
        "iterations": 2,
        "proof_steps": 1,
        "skip_plot": True
    }

    print("Testing gradient descent analysis...")
    returncode, stdout, stderr = run_mcp_command("analyze_gradient_descent", args)
    print(f"analyze_gradient_descent: {returncode}")
    if returncode == 0:
        print("✓ Gradient descent analysis works")
        return True
    else:
        print(f"✗ Failed: {stderr}")
        return False

def test_async_tools():
    """Test asynchronous job submission."""
    print("\nTesting asynchronous job submission...")

    # Submit a small job
    args = {
        "iterations": 3,
        "proof_steps": 2,
        "job_name": "test_job"
    }

    print("Submitting gradient descent job...")
    returncode, stdout, stderr = run_mcp_command("submit_gradient_descent_analysis", args)
    print(f"submit_gradient_descent_analysis: {returncode}")

    if returncode == 0:
        try:
            result = json.loads(stdout)
            job_id = result.get("job_id")
            print(f"✓ Job submitted with ID: {job_id}")

            # Wait a bit and check status
            time.sleep(2)
            status_args = {"job_id": job_id}
            returncode, stdout, stderr = run_mcp_command("get_job_status", status_args)

            if returncode == 0:
                status_result = json.loads(stdout)
                print(f"✓ Job status: {status_result.get('status', 'unknown')}")
                return True
            else:
                print(f"✗ Failed to get job status: {stderr}")
                return False

        except Exception as e:
            print(f"✗ Failed to parse job submission result: {e}")
            return False
    else:
        print(f"✗ Job submission failed: {stderr}")
        return False

def main():
    """Run all tests."""
    print("=== Testing MCP Server ===")

    # Test utility tools first
    pepflow_available = test_utility_tools()

    # Test job management
    test_job_management()

    if pepflow_available:
        # Test analysis tools if PEPFlow is available
        sync_works = test_sync_tools()
        if sync_works:
            test_async_tools()
    else:
        print("\nSkipping PEPFlow-dependent tests (framework not available)")

    print("\n=== Test Summary ===")
    print("MCP server basic functionality tested.")
    print("Use 'mamba run -p ./env python src/server.py' to start the server.")

if __name__ == "__main__":
    main()