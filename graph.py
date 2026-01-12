from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage

from state import StudyState
from agents import router_node, assistant_node, should_continue
from tools import all_tools


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
        Dictionary with response and detected subject
    """
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=message)],
        "current_subject": None,
        "session_id": session_id,
        "user_message": message,
        "needs_web_search": False
    }

    # Run the graph
    result = study_graph.invoke(initial_state)

    # Extract the final response
    final_message = result["messages"][-1]
    response_content = final_message.content if hasattr(final_message, "content") else str(final_message)

    return {
        "response": response_content,
        "detected_subject": result.get("current_subject"),
        "session_id": session_id
    }


# For visualization (optional)
def get_graph_image():
    """Generate a visualization of the graph (requires graphviz)."""
    try:
        return study_graph.get_graph().draw_mermaid_png()
    except Exception:
        return None
