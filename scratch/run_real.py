import asyncio
import json
import logging
import os
import sys

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.orchestrator import build_graph
from core.state import LessonPlanState

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
load_dotenv()

async def main():
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "test_run_real"}}
    
    state = LessonPlanState(
        subject="ELA",
        grade="5",
        topic="Analyze how setting, events, conflict, and characterization contribute to the plot in a literary text.",
        state="Florida",
        duration="45 minutes",
        lesson_date="2026-06-27",
        accommodations="",
        revision_count=0,
        max_revisions=3,
        review_passed=False,
    )
    
    print("Starting workflow...")
    # Run to interrupt
    result = await graph.ainvoke(state, config)
    
    # Confirm standard
    print("Approving standard...")
    result = await graph.ainvoke(Command(resume="Confirm standards selection"), config)
    
    # Wait for completion
    print("Generation complete. Final Markdown:")
    print("========================================")
    print(result.get("draft_lesson_plan", ""))
    print("========================================")
    
    with open("c:\\Projects\\Lesson Plan Helper Agent\\scratch\\output.md", "w", encoding="utf-8") as f:
        f.write(result.get("draft_lesson_plan", ""))

if __name__ == "__main__":
    asyncio.run(main())
