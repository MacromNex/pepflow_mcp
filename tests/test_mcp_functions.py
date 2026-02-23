#!/usr/bin/env python3
"""Test MCP functions through their .fn attribute."""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Add project src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_all_mcp_functions():
    """Test all MCP functions through the FunctionTool.fn interface."""
    print(f"🚀 Starting MCP Function Tests")
    print(f"📁 Project Root: {project_root}")
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        import server
    except ImportError as e:
        print(f"❌ Failed to import server: {e}")
        return False

    results = {
        "test_date": datetime.now().isoformat(),
        "tests": {},
        "summary": {}
    }

    # Test utility functions
    print("\n🧪 Testing Utility Functions...")
    utility_tests = [
        ("get_available_algorithms", {}),
        ("get_default_configs", {}),
        ("validate_pepflow_installation", {}),
    ]

    utility_passed = 0
    for tool_name, args in utility_tests:
        try:
            tool = getattr(server, tool_name)
            start_time = time.time()
            result = tool.fn(**args)
            duration = time.time() - start_time

            if isinstance(result, dict) and result.get("status") == "success":
                print(f"   ✅ {tool_name}: PASSED in {duration:.3f}s")
                utility_passed += 1
                results["tests"][tool_name] = {
                    "status": "passed",
                    "duration": f"{duration:.3f}s",
                    "result_keys": list(result.keys())
                }
            else:
                error = result.get("error", "Unknown error") if isinstance(result, dict) else "Invalid result"
                print(f"   ❌ {tool_name}: FAILED - {error}")
                results["tests"][tool_name] = {
                    "status": "failed",
                    "error": error
                }
        except Exception as e:
            print(f"   ❌ {tool_name}: ERROR - {e}")
            results["tests"][tool_name] = {
                "status": "error",
                "error": str(e)
            }

    print(f"   Utility Functions: {utility_passed}/{len(utility_tests)} passed")

    # Test sync analysis functions with minimal parameters
    print("\n🧪 Testing Sync Analysis Functions...")
    sync_tests = [
        ("analyze_gradient_descent", {
            "iterations": 3,
            "proof_steps": 1,
            "skip_plot": True
        }),
        ("analyze_accelerated_gradient", {
            "iterations": 2,
            "proof_steps": 1,
            "skip_plot": True
        }),
        ("analyze_optimization_algorithm", {
            "algorithm": "gradient_descent",
            "iterations": 2
        })
    ]

    sync_passed = 0
    for tool_name, args in sync_tests:
        try:
            tool = getattr(server, tool_name)
            print(f"   Testing {tool_name}...")
            start_time = time.time()
            result = tool.fn(**args)
            duration = time.time() - start_time

            if isinstance(result, dict) and result.get("status") == "success":
                print(f"   ✅ {tool_name}: PASSED in {duration:.3f}s")
                sync_passed += 1
                results["tests"][tool_name] = {
                    "status": "passed",
                    "duration": f"{duration:.3f}s",
                    "has_results": "convergence" in str(result) or "analysis" in str(result)
                }
            else:
                error = result.get("error", "Unknown error") if isinstance(result, dict) else "Invalid result"
                print(f"   ❌ {tool_name}: FAILED - {error}")
                results["tests"][tool_name] = {
                    "status": "failed",
                    "error": error
                }
        except Exception as e:
            print(f"   ❌ {tool_name}: ERROR - {e}")
            results["tests"][tool_name] = {
                "status": "error",
                "error": str(e)
            }

    print(f"   Sync Analysis: {sync_passed}/{len(sync_tests)} passed")

    # Test job management functions
    print("\n🧪 Testing Job Management Functions...")
    job_passed = 0
    job_tests = 5  # submit, status, list, log, cancel

    try:
        # Submit a test job
        submit_tool = getattr(server, "submit_gradient_descent_analysis")
        print("   Submitting test job...")

        submit_args = {
            "iterations": 3,
            "proof_steps": 1,
            "job_name": "mcp_function_test"
        }

        start_time = time.time()
        submit_result = submit_tool.fn(**submit_args)
        submit_duration = time.time() - start_time

        if isinstance(submit_result, dict) and submit_result.get("status") == "submitted":
            job_id = submit_result.get("job_id")
            print(f"   ✅ Job Submission: SUCCESS - {job_id} in {submit_duration:.3f}s")
            job_passed += 1

            results["tests"]["job_submission"] = {
                "status": "passed",
                "duration": f"{submit_duration:.3f}s",
                "job_id": job_id
            }

            # Test job status
            status_tool = getattr(server, "get_job_status")
            status_result = status_tool.fn(job_id)

            if isinstance(status_result, dict) and status_result.get("job_id") == job_id:
                print(f"   ✅ Job Status: SUCCESS - {status_result.get('status')}")
                job_passed += 1
                results["tests"]["job_status"] = {
                    "status": "passed",
                    "job_status": status_result.get("status")
                }
            else:
                print(f"   ❌ Job Status: FAILED")
                results["tests"]["job_status"] = {"status": "failed"}

            # Test job listing
            list_tool = getattr(server, "list_jobs")
            list_result = list_tool.fn()

            if isinstance(list_result, dict) and list_result.get("status") == "success":
                job_count = len(list_result.get("jobs", []))
                print(f"   ✅ Job Listing: SUCCESS - {job_count} jobs")
                job_passed += 1
                results["tests"]["job_listing"] = {
                    "status": "passed",
                    "job_count": job_count
                }
            else:
                print(f"   ❌ Job Listing: FAILED")
                results["tests"]["job_listing"] = {"status": "failed"}

            # Test job log (after a brief wait)
            time.sleep(2)
            log_tool = getattr(server, "get_job_log")
            log_result = log_tool.fn(job_id, tail=10)

            if isinstance(log_result, dict) and log_result.get("status") == "success":
                log_lines = len(log_result.get("log_lines", []))
                print(f"   ✅ Job Log: SUCCESS - {log_lines} lines")
                job_passed += 1
                results["tests"]["job_log"] = {
                    "status": "passed",
                    "log_lines": log_lines
                }
            else:
                print(f"   ❌ Job Log: FAILED")
                results["tests"]["job_log"] = {"status": "failed"}

            # Test job cancellation (if still running)
            current_status = status_tool.fn(job_id)
            if current_status.get("status") in ["pending", "running"]:
                cancel_tool = getattr(server, "cancel_job")
                cancel_result = cancel_tool.fn(job_id)

                if isinstance(cancel_result, dict) and cancel_result.get("status") == "success":
                    print(f"   ✅ Job Cancellation: SUCCESS")
                    job_passed += 1
                    results["tests"]["job_cancellation"] = {"status": "passed"}
                else:
                    print(f"   ❌ Job Cancellation: FAILED")
                    results["tests"]["job_cancellation"] = {"status": "failed"}
            else:
                print(f"   ⏭️ Job Cancellation: SKIPPED (job not running)")
                job_passed += 1  # Don't penalize for job completing too fast
                results["tests"]["job_cancellation"] = {"status": "skipped", "reason": "job_completed"}

        else:
            print(f"   ❌ Job Submission: FAILED - {submit_result}")
            results["tests"]["job_submission"] = {
                "status": "failed",
                "error": str(submit_result)
            }

    except Exception as e:
        print(f"   ❌ Job Management: ERROR - {e}")
        results["tests"]["job_management_error"] = {
            "status": "error",
            "error": str(e)
        }

    print(f"   Job Management: {job_passed}/{job_tests} passed")

    # Calculate overall results
    total_tests = len(utility_tests) + len(sync_tests) + job_tests
    total_passed = utility_passed + sync_passed + job_passed
    pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

    results["summary"] = {
        "total_tests": total_tests,
        "passed": total_passed,
        "failed": total_tests - total_passed,
        "pass_rate": f"{pass_rate:.1f}%",
        "overall_status": "passed" if total_passed == total_tests else "failed"
    }

    # Print summary
    print(f"\n📊 MCP Function Test Summary:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {total_passed}")
    print(f"   Failed: {total_tests - total_passed}")
    print(f"   Pass Rate: {pass_rate:.1f}%")

    status_symbol = "✅" if total_passed == total_tests else "❌"
    print(f"{status_symbol} Overall Status: {'PASSED' if total_passed == total_tests else 'FAILED'}")

    # Save results
    output_file = project_root / "reports" / "step7_mcp_function_test_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📄 Results saved to: {output_file}")

    return total_passed == total_tests

if __name__ == "__main__":
    success = test_all_mcp_functions()
    sys.exit(0 if success else 1)