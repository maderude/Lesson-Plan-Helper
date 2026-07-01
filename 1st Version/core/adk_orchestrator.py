import logging
import asyncio
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver

from google.adk.agents import Agent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset
from google.adk.tools.mcp_tool.stdio_params import StdioServerParams

from core.state import LessonPlanState

logger = logging.getLogger(__name__)

# ==============================================================================
# Global ADK Resources
# ==============================================================================
# We cache the tools and exit stack so we don't restart the MCP server on every run
_mcp_tools = None
_mcp_exit_stack = None

async def get_mcp_tools():
    global _mcp_tools, _mcp_exit_stack
    if _mcp_tools is None:
        import os
        mcp_server_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "server", "mcp_server.py")
        _mcp_tools, _mcp_exit_stack = await MCPToolset.from_server(
            connection_params=StdioServerParams(
                command="python",
                args=[mcp_server_path]
            )
        )
    return _mcp_tools

async def build_planner_agent():
    tools = await get_mcp_tools()
    return Agent(
        name="planner_agent",
        model="gemini-2.5-pro",
        tools=tools,
        instruction="""
        You are the Planner Agent for the Lesson Plan Helper.
        You have access to MCP tools and resources.
        Your task:
        1. Retrieve the drafting instructions using the `draft_lesson_plan` prompt.
        2. Draft a complete, high-quality lesson plan.
        3. Make sure to adhere to formatting guidelines from the formatting resource.
        """
    )

async def build_reviewer_agent():
    tools = await get_mcp_tools()
    return Agent(
        name="reviewer_agent",
        model="gemini-2.5-pro",
        tools=tools,
        instruction="""
        You are the Reviewer Agent for the Lesson Plan Helper.
        Your task is to review the lesson plan by calling the `score_lesson_rubric` tool.
        Return the exact output of that tool.
        """
    )

async def build_rewriter_agent():
    tools = await get_mcp_tools()
    return Agent(
        name="rewriter_agent",
        model="gemini-2.5-pro",
        tools=tools,
        instruction="""
        You are the Rewriter Agent for the Lesson Plan Helper.
        Your task is to fix the lesson plan by calling the `rewrite_with_feedback` tool.
        Return the exact output of that tool.
        """
    )

# ==============================================================================
# LangGraph Nodes
# ==============================================================================
async def intake_node(state: LessonPlanState) -> dict[str, Any]:
    # We could call `start_new_lesson_session` here, but for simplicity
    # we just pass the state through
    return {"status": "intake_complete"}

async def discovery_node(state: LessonPlanState) -> dict[str, Any]:
    tools = await get_mcp_tools()
    # Find the search_standards tool and call it
    search_tool = next((t for t in tools if t.name == "search_standards"), None)
    if search_tool:
        payload = {
            "grade": state["grade"],
            "subject": state["subject"],
            "state": state["state"],
            "topic": state["topic"]
        }
        res = await search_tool.execute(payload)
        # Mock setting retrieved standards based on what would come back
        return {"retrieved_standards": res}
    return {"errors": ["search_standards tool not found"]}

def confirmation_node(state: LessonPlanState) -> dict[str, Any]:
    interrupt("Confirm standards selection")
    return {"status": "standards_confirmed"}

async def planning_node(state: LessonPlanState) -> dict[str, Any]:
    agent = await build_planner_agent()
    # Invoke the agent to draft the plan
    # In a real app we'd pass the actual prompt
    response = await agent.run(f"Draft a lesson plan for topic {state['topic']} and grade {state['grade']}.")
    return {"draft_lesson_plan": response.content}

async def review_node(state: LessonPlanState) -> dict[str, Any]:
    agent = await build_reviewer_agent()
    # For now we mock the reviewer response logic to match our state
    return {"review_passed": True, "rubric_eval": {}}

async def rewrite_node(state: LessonPlanState) -> dict[str, Any]:
    agent = await build_rewriter_agent()
    return {"revision_count": state.get("revision_count", 0) + 1}

def route_after_review(state: LessonPlanState) -> str:
    if state.get("review_passed"):
        return END
    if state.get("revision_count", 0) >= 3:
        return END
    return "rewrite_node"

def build_graph(checkpointer=None) -> StateGraph:
    graph = StateGraph(LessonPlanState)
    graph.add_node("intake_node", intake_node)
    graph.add_node("discovery_node", discovery_node)
    graph.add_node("confirmation_node", confirmation_node)
    graph.add_node("planning_node", planning_node)
    graph.add_node("review_node", review_node)
    graph.add_node("rewrite_node", rewrite_node)
    
    graph.set_entry_point("intake_node")
    graph.add_edge("intake_node", "discovery_node")
    graph.add_edge("discovery_node", "confirmation_node")
    graph.add_edge("confirmation_node", "planning_node")
    graph.add_edge("planning_node", "review_node")
    graph.add_conditional_edges("review_node", route_after_review, {"rewrite_node": "rewrite_node", END: END})
    graph.add_edge("rewrite_node", "review_node")
    
    return graph.compile(checkpointer=checkpointer)
