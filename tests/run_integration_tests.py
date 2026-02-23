#!/usr/bin/env python3
"""Automated integration test runner for optimization-analysis MCP server."""

import json
import subprocess
import sys
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

class MCPTestRunner:
    """Test runner for MCP server integration validation."""

    def __init__(self, server_name: str = "optimization-analysis"):
        self.server_name = server_name
        self.project_root = Path(__file__).parent.parent
        self.results = {
            "test_date": datetime.now().isoformat(),
            "server_name": server_name,
            "project_root": str(self.project_root),
            "tests": {},
            "issues": [],
            "summary": {}
        }

    def log_result(self, test_name: str, status: str, details: Dict = None, error: str = None):
        """Log a test result."""
        self.results["tests"][test_name] = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details or {},
            "error": error
        }

        status_symbol = "✅" if status == "passed" else "❌" if status == "failed" else "⚠️"
        print(f"{status_symbol} {test_name}: {status.upper()}")
        if error:
            print(f"   Error: {error}")
        if details:
            for key, value in details.items():
                if isinstance(value, (str, int, float)):
                    print(f"   {key}: {value}")

    def run_command(self, cmd: List[str], timeout: int = 30, cwd: Optional[Path] = None) -> Dict[str, Any]:
        """Run a command and return results."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd or self.project_root
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "returncode": -1
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }

    def test_mcp_registration(self) -> bool:
        """Test that MCP server is registered in Claude Code."""
        print("\n🧪 Testing MCP Server Registration...")

        result = self.run_command(["claude", "mcp", "list"])

        if not result["success"]:
            self.log_result(
                "mcp_registration",
                "failed",
                error=f"Failed to run 'claude mcp list': {result['stderr']}"
            )
            return False

        if self.server_name in result["stdout"]:
            # Check for connection status
            if "✓ Connected" in result["stdout"]:
                self.log_result(
                    "mcp_registration",
                    "passed",
                    details={"connection_status": "connected"}
                )
                return True
            else:
                self.log_result(
                    "mcp_registration",
                    "failed",
                    error="Server registered but not connected"
                )
                return False
        else:
            self.log_result(
                "mcp_registration",
                "failed",
                error=f"Server '{self.server_name}' not found in MCP list"
            )
            return False

    def test_server_startup(self) -> bool:
        """Test that server starts without errors."""
        print("\n🧪 Testing Server Startup...")

        env_python = self.project_root / "env" / "bin" / "python"
        server_script = self.project_root / "src" / "server.py"

        if not env_python.exists():
            self.log_result(
                "server_startup",
                "failed",
                error=f"Python interpreter not found: {env_python}"
            )
            return False

        if not server_script.exists():
            self.log_result(
                "server_startup",
                "failed",
                error=f"Server script not found: {server_script}"
            )
            return False

        # Test server import and tool count
        cmd = [str(env_python), "-c", f"""
import sys
sys.path.insert(0, 'src')
try:
    from server import mcp
    import asyncio

    async def test_tools():
        tools = await mcp.get_tools()
        return len(tools)

    tool_count = asyncio.run(test_tools())
    print(f"SUCCESS: Server loaded with {{tool_count}} tools")
except Exception as e:
    print(f"ERROR: {{e}}")
    sys.exit(1)
"""]

        result = self.run_command(cmd, timeout=60)

        if result["success"] and "SUCCESS" in result["stdout"]:
            # Extract tool count
            try:
                tool_count = int(result["stdout"].split("tools")[0].split()[-1])
                self.log_result(
                    "server_startup",
                    "passed",
                    details={"tool_count": tool_count}
                )
                return True
            except:
                self.log_result(
                    "server_startup",
                    "passed",
                    details={"output": result["stdout"].strip()}
                )
                return True
        else:
            self.log_result(
                "server_startup",
                "failed",
                error=result["stderr"] or result["stdout"]
            )
            return False

    def test_pepflow_import(self) -> bool:
        """Test that PEPFlow framework is available."""
        print("\n🧪 Testing PEPFlow Framework Import...")

        env_python = self.project_root / "env" / "bin" / "python"
        cmd = [str(env_python), "-c", """
try:
    import pepflow as pf
    import sympy as sp
    import numpy as np
    import matplotlib.pyplot as plt
    print(f"SUCCESS: PEPFlow available, SymPy {sp.__version__}, NumPy {np.__version__}")
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)
"""]

        result = self.run_command(cmd)

        if result["success"] and "SUCCESS" in result["stdout"]:
            # Parse versions
            output = result["stdout"].strip()
            details = {"pepflow_status": "available"}
            if "SymPy" in output:
                parts = output.split()
                for i, part in enumerate(parts):
                    if part == "SymPy" and i+1 < len(parts):
                        details["sympy_version"] = parts[i+1].rstrip(",")
                    elif part == "NumPy" and i+1 < len(parts):
                        details["numpy_version"] = parts[i+1]

            self.log_result("pepflow_import", "passed", details=details)
            return True
        else:
            self.log_result(
                "pepflow_import",
                "failed",
                error=result["stderr"] or result["stdout"]
            )
            return False

    def test_job_directory(self) -> bool:
        """Test that job directory exists and is writable."""
        print("\n🧪 Testing Job Directory...")

        jobs_dir = self.project_root / "jobs"

        try:
            # Create jobs directory if it doesn't exist
            jobs_dir.mkdir(parents=True, exist_ok=True)

            # Test write access
            test_file = jobs_dir / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()  # Clean up

            self.log_result(
                "job_directory",
                "passed",
                details={"jobs_dir": str(jobs_dir), "writable": True}
            )
            return True
        except Exception as e:
            self.log_result(
                "job_directory",
                "failed",
                error=f"Job directory not writable: {e}"
            )
            return False

    def test_scripts_availability(self) -> bool:
        """Test that analysis scripts are available."""
        print("\n🧪 Testing Analysis Scripts...")

        scripts_dir = self.project_root / "scripts"
        required_scripts = [
            "gradient_descent_analysis.py",
            "accelerated_gradient_analysis.py",
            "pep_optimization_framework.py"
        ]

        missing_scripts = []
        for script in required_scripts:
            script_path = scripts_dir / script
            if not script_path.exists():
                missing_scripts.append(script)

        if missing_scripts:
            self.log_result(
                "scripts_availability",
                "failed",
                error=f"Missing scripts: {', '.join(missing_scripts)}"
            )
            return False
        else:
            self.log_result(
                "scripts_availability",
                "passed",
                details={"scripts_found": len(required_scripts)}
            )
            return True

    def test_configs_availability(self) -> bool:
        """Test that configuration files are available."""
        print("\n🧪 Testing Configuration Files...")

        configs_dir = self.project_root / "configs"
        required_configs = [
            "gradient_descent_config.json",
            "accelerated_gradient_config.json",
            "pep_optimization_config.json"
        ]

        missing_configs = []
        valid_configs = 0

        for config in required_configs:
            config_path = configs_dir / config
            if not config_path.exists():
                missing_configs.append(config)
            else:
                # Try to parse JSON
                try:
                    with open(config_path) as f:
                        json.load(f)
                    valid_configs += 1
                except json.JSONDecodeError:
                    missing_configs.append(f"{config} (invalid JSON)")

        if missing_configs:
            self.log_result(
                "configs_availability",
                "failed",
                error=f"Missing/invalid configs: {', '.join(missing_configs)}"
            )
            return False
        else:
            self.log_result(
                "configs_availability",
                "passed",
                details={"valid_configs": valid_configs}
            )
            return True

    def run_all_tests(self) -> bool:
        """Run all integration tests."""
        print(f"🚀 Starting Integration Tests for {self.server_name}")
        print(f"📁 Project Root: {self.project_root}")
        print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Define test methods in order
        tests = [
            ("MCP Registration", self.test_mcp_registration),
            ("Server Startup", self.test_server_startup),
            ("PEPFlow Import", self.test_pepflow_import),
            ("Job Directory", self.test_job_directory),
            ("Scripts Availability", self.test_scripts_availability),
            ("Configs Availability", self.test_configs_availability),
        ]

        # Run tests
        passed = 0
        failed = 0

        for test_name, test_method in tests:
            try:
                if test_method():
                    passed += 1
                else:
                    failed += 1
                    # Log as an issue
                    self.results["issues"].append({
                        "test": test_name,
                        "severity": "high",
                        "description": f"Test {test_name} failed"
                    })
            except Exception as e:
                failed += 1
                self.log_result(test_name.lower().replace(" ", "_"), "error", error=str(e))
                self.results["issues"].append({
                    "test": test_name,
                    "severity": "critical",
                    "description": f"Test {test_name} crashed: {e}"
                })

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
        print(f"\n📊 Test Summary:")
        print(f"   Total Tests: {total}")
        print(f"   Passed: {passed}")
        print(f"   Failed: {failed}")
        print(f"   Pass Rate: {pass_rate:.1f}%")

        status_symbol = "✅" if failed == 0 else "❌"
        print(f"{status_symbol} Overall Status: {'PASSED' if failed == 0 else 'FAILED'}")

        return failed == 0

    def generate_report(self) -> str:
        """Generate JSON report."""
        return json.dumps(self.results, indent=2)

    def save_report(self, output_file: Optional[Path] = None):
        """Save test report to file."""
        if output_file is None:
            output_file = self.project_root / "reports" / "step7_integration_test_results.json"

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write(self.generate_report())

        print(f"\n📄 Report saved to: {output_file}")

def main():
    """Main test runner."""
    runner = MCPTestRunner()
    success = runner.run_all_tests()
    runner.save_report()

    # Also create a markdown report
    markdown_report = create_markdown_report(runner.results)
    markdown_file = runner.project_root / "reports" / "step7_integration.md"
    with open(markdown_file, 'w') as f:
        f.write(markdown_report)
    print(f"📄 Markdown report saved to: {markdown_file}")

    return 0 if success else 1

def create_markdown_report(results: Dict) -> str:
    """Create a markdown version of the test report."""
    md = f"""# Step 7: MCP Integration Test Results

## Test Information
- **Test Date**: {results['test_date']}
- **Server Name**: {results['server_name']}
- **Project Root**: {results['project_root']}

## Summary

| Metric | Value |
|--------|-------|
| Total Tests | {results['summary']['total_tests']} |
| Passed | {results['summary']['passed']} |
| Failed | {results['summary']['failed']} |
| Pass Rate | {results['summary']['pass_rate']} |
| **Overall Status** | **{results['summary']['overall_status'].upper()}** |

## Detailed Results

| Test | Status | Details |
|------|--------|---------|
"""

    for test_name, test_result in results['tests'].items():
        status_symbol = "✅" if test_result['status'] == 'passed' else "❌"
        details = test_result.get('details', {})
        detail_str = ", ".join([f"{k}: {v}" for k, v in details.items()]) if details else "N/A"
        if test_result.get('error'):
            detail_str = f"ERROR: {test_result['error']}"

        md += f"| {test_name.replace('_', ' ').title()} | {status_symbol} {test_result['status'].title()} | {detail_str} |\n"

    if results['issues']:
        md += f"""
## Issues Found

| Issue | Severity | Description |
|-------|----------|-------------|
"""
        for issue in results['issues']:
            md += f"| {issue['test']} | {issue['severity'].upper()} | {issue['description']} |\n"

    md += f"""
## Next Steps

### If Tests Passed ✅
- Proceed with functional testing using test prompts
- Test sync tools, submit API, and job management
- Run end-to-end scenarios
- Document any performance observations

### If Tests Failed ❌
- Review failed tests in detail
- Check installation requirements
- Verify file permissions and paths
- Re-run tests after fixes

## Test Environment Requirements

- ✅ Claude Code installed with MCP support
- ✅ MCP server registered: `claude mcp list`
- ✅ PEPFlow framework installed in project environment
- ✅ All analysis scripts and configs available
- ✅ Job directory writable

---
*Generated on {results['test_date']}*
"""

    return md

if __name__ == "__main__":
    sys.exit(main())