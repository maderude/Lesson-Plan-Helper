import sys
sys.path.append("c:\\Projects\\Lesson Plan Helper Agent")
from core.orchestrator import build_graph, planning_node
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

try:
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "test_thread_1"}}
    inputs = {
        "subject": "ELA",
        "grade": "5",
        "topic": "main idea and supporting details",
        "state": "Florida",
        "duration": "45 minutes",
        "lesson_date": "2025-09-15",
    }
    print("Running graph to interrupt...")
    for event in graph.stream(inputs, config=config):
        print(event)
        
    print("Resuming graph...")
    resume_payload = {"approved": True, "override_code": "ELA.5.R.1.2", "override_text": "text"}
    for event in graph.stream(Command(resume=resume_payload), config=config):
        print(event)
        
except Exception as e:
    import traceback
    traceback.print_exc()
