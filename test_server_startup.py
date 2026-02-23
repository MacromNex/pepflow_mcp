#!/usr/bin/env python3
"""Test MCP server startup without hanging."""

import sys
import time
import subprocess
from pathlib import Path

def test_server_startup():
    """Test that the MCP server starts without errors."""
    project_root = Path(__file__).parent
    server_script = project_root / "src" / "server.py"
    env_python = project_root / "env" / "bin" / "python"

    print(f"Testing server startup: {server_script}")
    print(f"Using Python: {env_python}")

    try:
        # Test basic import and instantiation
        cmd = [str(env_python), "-c", """
import sys
sys.path.insert(0, 'src')
from server import mcp
import asyncio

async def test_tools():
    tools = await mcp.get_tools()
    return len(tools)

tool_count = asyncio.run(test_tools())
print(f'✅ Server instantiated: {mcp.name}')
print(f'✅ Found {tool_count} registered tools')
"""]

        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print("✅ Server startup test PASSED")
            print("Server output:")
            for line in result.stdout.strip().split('\n'):
                print(f"  {line}")
            return True
        else:
            print("❌ Server startup test FAILED")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("❌ Server startup test TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ Server startup test ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_server_startup()
    sys.exit(0 if success else 1)