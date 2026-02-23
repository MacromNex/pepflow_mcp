#!/usr/bin/env python3
"""
Simple test for MCP server tools.
"""

import sys
sys.path.append('src')
sys.path.append('scripts')

def test_server_import():
    """Test that the server imports correctly."""
    try:
        from src.server import mcp
        print("✓ MCP server imports successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import MCP server: {e}")
        return False

def test_validation():
    """Test validation function."""
    try:
        # Import and test validation function directly
        from src.server import validate_pepflow_installation
        result = validate_pepflow_installation()

        if result.get('status') == 'success':
            print("✓ PEPFlow validation successful")
            return True
        else:
            print(f"✗ PEPFlow validation failed: {result}")
            return False
    except Exception as e:
        print(f"✗ Validation test failed: {e}")
        return False

def test_algorithm_list():
    """Test algorithm listing function."""
    try:
        from src.server import get_available_algorithms
        result = get_available_algorithms()

        if result.get('status') == 'success':
            algorithms = result.get('algorithms', {})
            print(f"✓ Found {len(algorithms)} algorithms: {list(algorithms.keys())}")
            return True
        else:
            print(f"✗ Algorithm listing failed: {result}")
            return False
    except Exception as e:
        print(f"✗ Algorithm listing test failed: {e}")
        return False

def test_sync_analysis():
    """Test synchronous analysis function."""
    try:
        from src.server import analyze_gradient_descent
        result = analyze_gradient_descent(
            iterations=3,
            proof_steps=2,
            skip_plot=True,
            output_dir='results/test_mcp_sync'
        )

        if result.get('status') == 'success':
            print("✓ Synchronous gradient descent analysis successful")
            return True
        else:
            print(f"✗ Synchronous analysis failed: {result}")
            return False
    except Exception as e:
        print(f"✗ Synchronous analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_job_submission():
    """Test job submission function."""
    try:
        from src.server import submit_gradient_descent_analysis
        result = submit_gradient_descent_analysis(
            iterations=3,
            proof_steps=2,
            job_name="test_mcp_job",
            output_dir="results/test_mcp_async"
        )

        if result.get('status') == 'submitted':
            job_id = result.get('job_id')
            print(f"✓ Job submission successful, job_id: {job_id}")

            # Test job status
            import time
            time.sleep(1)

            from src.server import get_job_status
            status_result = get_job_status(job_id)
            print(f"✓ Job status: {status_result.get('status', 'unknown')}")

            return True
        else:
            print(f"✗ Job submission failed: {result}")
            return False
    except Exception as e:
        print(f"✗ Job submission test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=== Testing MCP Server Components ===")

    tests = [
        ("Server Import", test_server_import),
        ("PEPFlow Validation", test_validation),
        ("Algorithm Listing", test_algorithm_list),
        ("Sync Analysis", test_sync_analysis),
        ("Job Submission", test_job_submission)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1

    print(f"\n=== Test Summary ===")
    print(f"Passed: {passed}/{total} tests")

    if passed == total:
        print("🎉 All tests passed! MCP server is ready to use.")
    else:
        print("⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()