import asyncio
import json
from core.orchestrator import get_mcp_tools

async def test():
    tools = await get_mcp_tools()
    ts_tools = await tools[0].get_tools()
    t = next(t for t in ts_tools if t.name == 'validate_lesson_structure')
    res = await t.run_async(args={'payload': {'lesson_markdown': 'something'}}, tool_context=None)
    print("TYPE IS:", type(res))
    print("RES IS:", res)
    if isinstance(res, dict) and "content" in res and res["content"]:
        val_dict = json.loads(res["content"][0]["text"])
        print("VAL DICT IS:", val_dict)

asyncio.run(test())
