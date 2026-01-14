from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage

from state import StudyState
from agents import (
    router_node, assistant_node, should_continue,
    rewrite_to_concise, get_llm, MAX_RESPONSE_LENGTH
)
from tools import all_tools

# Session store to maintain state across conversation turns
# This preserves current_subject, last_subject_question, etc. between messages
_session_store: dict = {}


def create_study_graph():
    """
    Create the LangGraph workflow for the study platform.

    Flow:
    1. User message comes in
    2. Router classifies the subject
    3. Assistant responds (may call tools)
    4. If tools called, execute them and loop back to assistant
    5. Return final response
    """

    # Create the graph
    workflow = StateGraph(StudyState)

    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("assistant", assistant_node)
    workflow.add_node("tools", ToolNode(all_tools))

    # Set entry point
    workflow.set_entry_point("router")

    # Add edges
    # Router always goes to assistant
    workflow.add_edge("router", "assistant")

    # Assistant conditionally goes to tools or ends
    workflow.add_conditional_edges(
        "assistant",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )

    # Tools always go back to assistant
    workflow.add_edge("tools", "assistant")

    # Compile the graph
    graph = workflow.compile()

    return graph


# Create the compiled graph
study_graph = create_study_graph()


def chat(message: str, session_id: str = "default") -> dict:
    """
    Process a chat message through the study platform.

    Args:
        message: The user's message
        session_id: Session identifier for conversation tracking

    Returns:
        Dictionary with response, detected subject, style, and session_id
    """
    global _session_store

    # Get existing session state or initialize new one
    session_state = _session_store.get(session_id, {
        "messages": [],
        "current_subject": None,
        "last_subject_question": None,
        "user_level": "beginner",
        "style": "normal"
    })

    # Build state for this turn, preserving cross-turn context
    initial_state = {
        "messages": session_state["messages"] + [HumanMessage(content=message)],
        "current_subject": session_state["current_subject"],  # Preserve from previous turn
        "session_id": session_id,
        "user_message": message,
        "needs_web_search": False,
        "verbose_mode": False,
        "user_level": session_state.get("user_level", "beginner"),
        "style": "normal",  # Reset style each turn (router will set to "simple" if needed)
        "last_subject_question": session_state.get("last_subject_question")
    }

    # Run the graph
    result = study_graph.invoke(initial_state)

    # Extract the final response
    final_message = result["messages"][-1]
    response_content = final_message.content if hasattr(final_message, "content") else str(final_message)

    # Post-processing: auto-rewrite long responses if not in verbose mode (skip for simple mode)
    verbose_mode = result.get("verbose_mode", False)
    style = result.get("style", "normal")
    if not verbose_mode and style != "simple" and len(response_content) > MAX_RESPONSE_LENGTH:
        llm = get_llm()
        response_content = rewrite_to_concise(response_content, llm)

    # Update session store with new state for next turn
    _session_store[session_id] = {
        "messages": result["messages"],
        "current_subject": result.get("current_subject"),
        "last_subject_question": result.get("last_subject_question"),
        "user_level": result.get("user_level", "beginner"),
        "style": result.get("style", "normal")
    }

    return {
        "response": response_content,
        "detected_subject": result.get("current_subject"),
        "style": result.get("style", "normal"),
        "session_id": session_id
    }


def clear_session(session_id: str = "default") -> None:
    """Clear session state for a given session ID."""
    global _session_store
    if session_id in _session_store:
        del _session_store[session_id]


# For visualization (optional)
def get_graph_image():
    """Generate a visualization of the graph (requires graphviz)."""
    try:
        return study_graph.get_graph().draw_mermaid_png()
    except Exception:
        return None
