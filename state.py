from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

from models import Subject


class StudyState(TypedDict):
    """State for the study platform graph."""

    # Messages in the conversation (using LangGraph's message handling)
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
