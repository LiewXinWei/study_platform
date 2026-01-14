from typing import Annotated, List, Optional, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

from models import Subject


class StudyState(TypedDict):
    """State for the study platform graph.

    IMPORTANT - Message Reducer Pattern:
    The `messages` field uses `Annotated[list, add_messages]` which applies
    the add_messages reducer. This is CRITICAL because:

    1. Without a reducer, LangGraph uses "last-write-wins" semantics.
       If two nodes run in parallel and both write to `messages`, one
       overwrites the other and you LOSE data.

    2. The add_messages reducer APPENDS new messages instead of replacing.
       This ensures tool outputs, assistant responses, and human messages
       all accumulate correctly in the conversation history.

    3. This is especially important when:
       - Tools execute and return results (ToolMessage)
       - Multiple nodes produce outputs in the same graph execution
       - You need deterministic, reproducible conversation state

    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers
    """

    # Messages in the conversation - uses add_messages reducer to APPEND not overwrite
    # Without this reducer, parallel node writes would cause last-write-wins data loss
    messages: Annotated[list, add_messages]

    # The detected subject for the current message
    current_subject: Optional[Subject]

    # Session ID for tracking conversation
    session_id: str

    # The user's original message (for reference)
    user_message: str

    # Flag to indicate if we should use web search
    needs_web_search: bool

    # Flag to indicate verbose mode (user requested detail)
    verbose_mode: bool

    # User's experience level: "beginner" (default) or "advanced"
    user_level: str

    # Response style: "normal" or "simple" (triggered by SIMPLIFY intent)
    style: str

    # Store the last meaningful subject question for "explain that" follow-ups
    last_subject_question: Optional[str]

    # Store the last assistant answer for "what do you mean" follow-ups
    last_assistant_answer: Optional[str]

    # Track clarification attempts to avoid infinite loops
    attempts_clarify: int

    # Router decision result (JSON structure with mode, confidence, etc.)
    router_result: Optional[dict]

    # Flag to indicate if verifier passed (used for conditional routing)
    verifier_passed: bool

    # Verifier feedback if revision needed
    verifier_feedback: Optional[str]
