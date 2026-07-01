import sys
sys.path.append("c:\\Projects\\Lesson Plan Helper Agent")
from core.orchestrator import build_graph, planning_node
import inspect

print(f"planning_node iscoroutinefunction: {inspect.iscoroutinefunction(planning_node)}")

try:
    graph = build_graph()
    print("Graph compiled successfully.")
    
    # Run a quick test
    inputs = {
        "subject": "ELA",
        "grade": "5",
        "topic": "main idea and supporting details",
        "state": "Florida",
        "duration": "45 minutes",
        "lesson_date": "2025-09-15",
        "selected_standard_code": "ELA.5.R.1.2",
        "selected_standard_text": "Explain how relevant details support the central idea...",
    }
    print("Running graph...")
    # Jump straight to planning node since discovery/intake usually require input
    for event in graph.stream(inputs):
        print(event)
    
except Exception as e:
    import traceback
    traceback.print_exc()
