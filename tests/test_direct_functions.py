#!/usr/bin/env python3
"""Direct function testing - bypasses MCP interface to test actual functions."""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add project src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_utility_functions():
    """Test utility functions directly."""
    print("\n🧪 Testing Utility Functions...")

    try:
        from server import (
            get_available_algorithms,
            get_default_configs,
            validate_pepflow_installation
        )

        tests = [
            ("get_available_algorithms", get_available_algorithms, {}),
            ("get_default_configs", get_default_configs, {}),
            ("validate_pepflow_installation", validate_pepflow_installation, {})
        ]

        passed = 0
        for name, func, args in tests:
            try:
                start_time = time.time()
                result = func(**args)
                duration = time.time() - start_time

                if isinstance(result, dict) and result.get("status") == "success":
                    print(f"   ✅ {name}: PASSED in {duration:.3f}s")
                    passed += 1
                else:
                    print(f"   ❌ {name}: FAILED - {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"   ❌ {name}: ERROR - {e}")

        print(f"   Utility Functions: {passed}/{len(tests)} passed")
        return passed == len(tests)

    except ImportError as e:
        print(f"   ❌ Import Error: {e}")
        return False

def test_sync_analysis_functions():
    """Test sync analysis functions directly."""
    print("\n🧪 Testing Sync Analysis Functions...")

    try:
        from server import (
            analyze_gradient_descent,
            analyze_accelerated_gradient,
            analyze_optimization_algorithm
        )

        tests = [
            ("analyze_gradient_descent", analyze_gradient_descent, {
                "iterations": 3,
                "proof_steps": 1,
                "skip_plot": True
            }),
            ("analyze_accelerated_gradient", analyze_accelerated_gradient, {
                "iterations": 2,
                "proof_steps": 1,
                "skip_plot": True
            }),
            ("analyze_optimization_algorithm", analyze_optimization_algorithm, {
                "algorithm": "gradient_descent",
                "iterations": 2
            })
        ]

        passed = 0
        for name, func, args in tests:
            try:
                print(f"   Testing {name}...")
                start_time = time.time()
                result = func(**args)
                duration = time.time() - start_time

                if isinstance(result, dict) and result.get("status") == "success":
                    print(f"   ✅ {name}: PASSED in {duration:.3f}s")
                    passed += 1
                else:
                    error = result.get("error", "Unknown error") if isinstance(result, dict) else "Invalid result format"
                    print(f"   ❌ {name}: FAILED - {error}")
            except Exception as e:
                print(f"   ❌ {name}: ERROR - {e}")

        print(f"   Sync Analysis: {passed}/{len(tests)} passed")
        return passed == len(tests)

    except ImportError as e:
        print(f"   ❌ Import Error: {e}")
        return False

def test_job_functions():
    """Test job management functions directly."""
    print("\n🧪 Testing Job Management Functions...")

    try:
        from server import (
            submit_gradient_descent_analysis,
            get_job_status,
            list_jobs,
            cancel_job
        )

        # Submit a test job
        print("   Submitting test job...")
        submit_args = {
            "iterations": 3,
            "proof_steps": 1,
            "job_name": "direct_test_job"
        }

        start_time = time.time()
        result = submit_gradient_descent_analysis(**submit_args)
        submit_duration = time.time() - start_time

        if not isinstance(result, dict) or result.get("status") != "submitted":
            print(f"   ❌ Job Submission: FAILED - {result}")
            return False

        job_id = result.get("job_id")
        if not job_id:
            print("   ❌ Job Submission: No job ID returned")
            return False

        print(f"   ✅ Job submitted: {job_id} in {submit_duration:.3f}s")

        # Test job status
        print("   Checking job status...")
        status_result = get_job_status(job_id)

        if isinstance(status_result, dict) and status_result.get("job_id") == job_id:
            print(f"   ✅ Job Status: {status_result.get('status')}")
        else:
            print(f"   ❌ Job Status: FAILED - {status_result}")
            return False

        # Test job listing
        print("   Testing job listing...")
        list_result = list_jobs()

        if isinstance(list_result, dict) and list_result.get("status") == "success":
            job_count = len(list_result.get("jobs", []))
            print(f"   ✅ Job Listing: Found {job_count} jobs")
        else:
            print(f"   ❌ Job Listing: FAILED - {list_result}")
            return False

        # Wait a moment then try to cancel the job (if it's still running)
        time.sleep(2)
        final_status = get_job_status(job_id)
        if final_status.get("status") in ["pending", "running"]:
            print("   Testing job cancellation...")
            cancel_result = cancel_job(job_id)
            if isinstance(cancel_result, dict) and cancel_result.get("status") == "success":
                print(f"   ✅ Job Cancellation: SUCCESS")
            else:
                print(f"   ⚠️ Job Cancellation: {cancel_result}")

        print("   Job Management: ALL TESTS PASSED")
        return True

    except ImportError as e:
        print(f"   ❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Job Management Error: {e}")
        return False

def main():
    """Run all direct function tests."""
    print(f"🚀 Starting Direct Function Tests")
    print(f"📁 Project Root: {project_root}")
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    tests = [
        ("Utility Functions", test_utility_functions),
        ("Sync Analysis Functions", test_sync_analysis_functions),
        ("Job Management Functions", test_job_functions)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name}: CRASHED - {e}")
            failed += 1

    # Print summary
    total = passed + failed
    pass_rate = (passed / total * 100) if total > 0 else 0

    print(f"\n📊 Direct Function Test Summary:")
    print(f"   Total Tests: {total}")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print(f"   Pass Rate: {pass_rate:.1f}%")

    status_symbol = "✅" if failed == 0 else "❌"
    print(f"{status_symbol} Overall Status: {'PASSED' if failed == 0 else 'FAILED'}")

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())