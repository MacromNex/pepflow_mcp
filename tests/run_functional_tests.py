#!/usr/bin/env python3
"""Functional test runner for MCP tools - tests actual tool execution."""

import json
import time
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

class MCPFunctionalTester:
    """Test actual MCP tool functionality."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        sys.path.insert(0, str(self.project_root / "src"))

        # Import the server
        try:
            from server import mcp
            self.mcp = mcp
        except ImportError as e:
            print(f"❌ Failed to import MCP server: {e}")
            sys.exit(1)

        self.results = {
            "test_date": datetime.now().isoformat(),
            "tests": {},
            "summary": {}
        }

    async def test_tool_discovery(self):
        """Test that tools can be discovered and listed."""
        print("\n🧪 Testing Tool Discovery...")

        try:
            start_time = time.time()
            tools = await self.mcp.get_tools()
            duration = time.time() - start_time

            tool_names = list(tools.keys())
            tool_count = len(tool_names)

            # Check for expected tool categories
            job_mgmt_tools = [name for name in tool_names if name.startswith(('get_job', 'cancel_job', 'list_jobs'))]
            sync_tools = [name for name in tool_names if name.startswith('analyze_')]
            submit_tools = [name for name in tool_names if name.startswith('submit_')]
            utility_tools = [name for name in tool_names if name.startswith(('get_', 'validate_'))]

            self.results["tests"]["tool_discovery"] = {
                "status": "passed",
                "duration": f"{duration:.3f}s",
                "details": {
                    "total_tools": tool_count,
                    "tool_names": tool_names,
                    "job_management_tools": len(job_mgmt_tools),
                    "sync_analysis_tools": len(sync_tools),
                    "submit_tools": len(submit_tools),
                    "utility_tools": len(utility_tools)
                }
            }

            print(f"✅ Tool Discovery: Found {tool_count} tools in {duration:.3f}s")
            print(f"   Categories: {len(job_mgmt_tools)} job mgmt, {len(sync_tools)} sync, {len(submit_tools)} submit, {len(utility_tools)} utility")
            return True

        except Exception as e:
            self.results["tests"]["tool_discovery"] = {
                "status": "failed",
                "error": str(e)
            }
            print(f"❌ Tool Discovery: FAILED - {e}")
            return False

    async def test_utility_tools(self):
        """Test utility tools that don't require heavy computation."""
        print("\n🧪 Testing Utility Tools...")

        utility_tests = [
            ("get_available_algorithms", {}),
            ("get_default_configs", {}),
            ("validate_pepflow_installation", {})
        ]

        passed = 0
        total = len(utility_tests)

        for tool_name, args in utility_tests:
            try:
                tools = await self.mcp.get_tools()
                if tool_name not in tools:
                    print(f"❌ {tool_name}: Tool not found")
                    continue

                start_time = time.time()

                # Import and call the function directly
                from server import (
                    get_available_algorithms,
                    get_default_configs,
                    validate_pepflow_installation
                )

                if tool_name == "get_available_algorithms":
                    result = get_available_algorithms(**args)
                elif tool_name == "get_default_configs":
                    result = get_default_configs(**args)
                elif tool_name == "validate_pepflow_installation":
                    result = validate_pepflow_installation(**args)
                else:
                    result = {"status": "error", "error": f"Unknown tool: {tool_name}"}

                duration = time.time() - start_time

                # Check result structure
                if isinstance(result, dict) and result.get("status") == "success":
                    print(f"✅ {tool_name}: PASSED in {duration:.3f}s")
                    passed += 1

                    self.results["tests"][tool_name] = {
                        "status": "passed",
                        "duration": f"{duration:.3f}s",
                        "result_keys": list(result.keys()) if isinstance(result, dict) else None
                    }
                else:
                    print(f"❌ {tool_name}: Unexpected result format")
                    self.results["tests"][tool_name] = {
                        "status": "failed",
                        "error": "Unexpected result format",
                        "result": str(result)[:100]
                    }

            except Exception as e:
                print(f"❌ {tool_name}: ERROR - {e}")
                self.results["tests"][tool_name] = {
                    "status": "error",
                    "error": str(e)
                }

        print(f"   Utility Tools: {passed}/{total} passed")
        return passed == total

    async def test_sync_analysis_tools(self):
        """Test synchronous analysis tools with minimal parameters."""
        print("\n🧪 Testing Sync Analysis Tools...")

        # Use small parameters for quick testing
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

        passed = 0
        total = len(sync_tests)

        for tool_name, args in sync_tests:
            try:
                tools = await self.mcp.get_tools()
                if tool_name not in tools:
                    print(f"❌ {tool_name}: Tool not found")
                    continue

                print(f"   Testing {tool_name}...")
                start_time = time.time()

                # Import and call the function directly from the server module
                from server import (
                    analyze_gradient_descent,
                    analyze_accelerated_gradient,
                    analyze_optimization_algorithm
                )

                if tool_name == "analyze_gradient_descent":
                    result = analyze_gradient_descent(**args)
                elif tool_name == "analyze_accelerated_gradient":
                    result = analyze_accelerated_gradient(**args)
                elif tool_name == "analyze_optimization_algorithm":
                    result = analyze_optimization_algorithm(**args)
                else:
                    result = {"status": "error", "error": f"Unknown tool: {tool_name}"}

                duration = time.time() - start_time

                # Check result structure
                if isinstance(result, dict) and result.get("status") == "success":
                    print(f"   ✅ {tool_name}: PASSED in {duration:.3f}s")
                    passed += 1

                    self.results["tests"][tool_name] = {
                        "status": "passed",
                        "duration": f"{duration:.3f}s",
                        "result_keys": list(result.keys()) if isinstance(result, dict) else None
                    }
                else:
                    print(f"   ❌ {tool_name}: Analysis failed")
                    self.results["tests"][tool_name] = {
                        "status": "failed",
                        "error": result.get("error", "Unknown error") if isinstance(result, dict) else "Invalid result format"
                    }

            except Exception as e:
                print(f"   ❌ {tool_name}: ERROR - {e}")
                self.results["tests"][tool_name] = {
                    "status": "error",
                    "error": str(e)
                }

        print(f"   Sync Analysis: {passed}/{total} passed")
        return passed == total

    async def test_job_submission(self):
        """Test job submission and management."""
        print("\n🧪 Testing Job Submission...")

        try:
            from server import submit_gradient_descent_analysis, get_job_status, list_jobs

            # Submit a quick job
            submit_args = {
                "iterations": 3,
                "proof_steps": 1,
                "job_name": "test_job_functional"
            }

            print("   Submitting test job...")
            start_time = time.time()
            result = submit_gradient_descent_analysis(**submit_args)
            submit_duration = time.time() - start_time

            if not isinstance(result, dict) or result.get("status") != "submitted":
                print(f"   ❌ Job Submission: FAILED - {result}")
                self.results["tests"]["job_submission"] = {
                    "status": "failed",
                    "error": "Job submission failed"
                }
                return False

            job_id = result.get("job_id")
            if not job_id:
                print("   ❌ Job Submission: No job ID returned")
                self.results["tests"]["job_submission"] = {
                    "status": "failed",
                    "error": "No job ID returned"
                }
                return False

            print(f"   ✅ Job submitted: {job_id} in {submit_duration:.3f}s")

            # Test job status
            print("   Checking job status...")
            status_result = get_job_status(job_id)

            if isinstance(status_result, dict) and status_result.get("job_id") == job_id:
                print(f"   ✅ Job Status: {status_result.get('status')}")

                # Test job listing
                print("   Testing job listing...")
                list_result = list_jobs()

                if isinstance(list_result, dict) and list_result.get("status") == "success":
                    job_count = len(list_result.get("jobs", []))
                    print(f"   ✅ Job Listing: Found {job_count} jobs")

                    self.results["tests"]["job_submission"] = {
                        "status": "passed",
                        "details": {
                            "job_id": job_id,
                            "submit_duration": f"{submit_duration:.3f}s",
                            "job_status": status_result.get("status"),
                            "total_jobs_listed": job_count
                        }
                    }
                    return True
                else:
                    print(f"   ❌ Job Listing: FAILED - {list_result}")
                    self.results["tests"]["job_submission"] = {
                        "status": "failed",
                        "error": "Job listing failed"
                    }
                    return False
            else:
                print(f"   ❌ Job Status: FAILED - {status_result}")
                self.results["tests"]["job_submission"] = {
                    "status": "failed",
                    "error": "Job status check failed"
                }
                return False

        except Exception as e:
            print(f"   ❌ Job Submission: ERROR - {e}")
            self.results["tests"]["job_submission"] = {
                "status": "error",
                "error": str(e)
            }
            return False

    async def run_all_functional_tests(self):
        """Run all functional tests."""
        print(f"🚀 Starting Functional Tests")
        print(f"📁 Project Root: {self.project_root}")
        print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        tests = [
            ("Tool Discovery", self.test_tool_discovery),
            ("Utility Tools", self.test_utility_tools),
            ("Sync Analysis Tools", self.test_sync_analysis_tools),
            ("Job Submission", self.test_job_submission)
        ]

        passed = 0
        failed = 0

        for test_name, test_method in tests:
            try:
                if await test_method():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"❌ {test_name}: CRASHED - {e}")
                failed += 1
                self.results["tests"][test_name.lower().replace(" ", "_")] = {
                    "status": "error",
                    "error": str(e)
                }

        # Generate summary
        total = passed + failed
        pass_rate = (passed / total * 100) if total > 0 else 0

        self.results["summary"] = {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{pass_rate:.1f}%",
            "overall_status": "passed" if failed == 0 else "failed"
        }

        # Print summary
        print(f"\n📊 Functional Test Summary:")
        print(f"   Total Tests: {total}")
        print(f"   Passed: {passed}")
        print(f"   Failed: {failed}")
        print(f"   Pass Rate: {pass_rate:.1f}%")

        status_symbol = "✅" if failed == 0 else "❌"
        print(f"{status_symbol} Overall Status: {'PASSED' if failed == 0 else 'FAILED'}")

        return failed == 0

    def save_report(self):
        """Save functional test report."""
        output_file = self.project_root / "reports" / "step7_functional_test_results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n📄 Functional test report saved to: {output_file}")

async def main():
    """Main functional test runner."""
    tester = MCPFunctionalTester()
    success = await tester.run_all_functional_tests()
    tester.save_report()

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))