import asyncio
from core.orchestrator import get_mcp_tools

async def test():
    tools = await get_mcp_tools()
    ts_tools = await tools[0].get_tools()
    print([t.name for t in ts_tools])

asyncio.run(test())
