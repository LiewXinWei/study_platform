from enum import Enum
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class Subject(str, Enum):
    """Supported subjects for the study platform."""
    PYTHON = "python"
    LANGGRAPH = "langgraph"
    LANGCHAIN = "langchain"
    JAVASCRIPT = "javascript"
    LLM = "llm"
    AUTOMATION = "automation"
    N8N = "n8n"
    GOHIGHLEVEL = "gohighlevel"
    GENERAL = "general"  # Fallback for unclassified messages


class Note(BaseModel):
    """A study note for a subject."""
    id: str
    subject: Subject
    content: str
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


class Solution(BaseModel):
    """A problem-solution pair from past experience."""
    id: str
    subject: Subject
    problem: str
    solution: str
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)


class Message(BaseModel):
    """A chat message."""
    role: str  # "user" or "assistant"
    content: str
    subject: Optional[Subject] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# API Request/Response Models

class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    message: str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    """Response body for chat endpoint."""
    response: str
    detected_subject: Subject
    session_id: str


class NoteRequest(BaseModel):
    """Request body for creating a note."""
    content: str
    tags: List[str] = Field(default_factory=list)


class NoteResponse(BaseModel):
    """Response body for note operations."""
    note: Note
    message: str


class SolutionRequest(BaseModel):
    """Request body for creating a solution."""
    problem: str
    solution: str
    tags: List[str] = Field(default_factory=list)


class SolutionResponse(BaseModel):
    """Response body for solution operations."""
    solution: Solution
    message: str


class SubjectListResponse(BaseModel):
    """Response body for listing subjects."""
    subjects: List[str]
    count: int
