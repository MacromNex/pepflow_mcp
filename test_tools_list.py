#!/usr/bin/env python3
"""Test script to list MCP tools."""

import asyncio
from src.server import mcp

async def test_tools():
    """Test that tools are properly registered."""
    try:
        tools = await mcp.get_tools()
        print(f'Found {len(tools)} tools:')
        for tool_name, tool in tools.items():
            print(f'  - {tool_name}: {tool.description[:80]}...' if len(tool.description) > 80 else f'  - {tool_name}: {tool.description}')

        print(f'\nServer name: {mcp.name}')
        print("All tools loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading tools: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_tools())
    exit(0 if success else 1)