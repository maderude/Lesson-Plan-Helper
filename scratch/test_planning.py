import asyncio
from core.orchestrator import planning_node
async def test():
    state = {'topic': 'verbs', 'grade': '5', 'subject': 'ELA', 'duration': '45', 'selected_standard_code': 'ELA', 'selected_standard_text': 'verbs', 'accommodations': 'none'}
    res = await planning_node(state)
    print(res['draft_lesson_plan'])
asyncio.run(test())
