import os
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file
load_dotenv()

from models import (
    Subject,
    ChatRequest,
    ChatResponse,
    NoteRequest,
    NoteResponse,
    SolutionRequest,
    SolutionResponse,
    SubjectListResponse,
    Note,
    Solution,
    Message
)
from storage import storage
from graph import chat


# Create FastAPI app
app = FastAPI(
    title="Study Platform API",
    description="Multi-subject chatbot platform for learning with LangGraph",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== Chat Endpoints ==========

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def send_message(request: ChatRequest):
    """
    Send a message to the study assistant.
    The system will automatically detect the subject and route to the appropriate assistant.
    """
    try:
        result = chat(request.message, request.session_id or "default")

        # Store the conversation in history
        subject = result.get("detected_subject", Subject.GENERAL)
        storage.add_message(
            request.session_id or "default",
            subject,
            "user",
            request.message
        )
        storage.add_message(
            request.session_id or "default",
            subject,
            "assistant",
            result["response"]
        )

        return ChatResponse(
            response=result["response"],
            detected_subject=subject,
            session_id=request.session_id or "default"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== Subject Endpoints ==========

@app.get("/subjects", response_model=SubjectListResponse, tags=["Subjects"])
async def list_subjects():
    """List all available subjects."""
    subjects = [s.value for s in Subject if s != Subject.GENERAL]
    return SubjectListResponse(subjects=subjects, count=len(subjects))


# ========== Notes Endpoints ==========

@app.get("/notes/{subject}", response_model=List[Note], tags=["Notes"])
async def get_notes(subject: str):
    """Get all notes for a specific subject."""
    try:
        subj = Subject(subject.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid subject: {subject}")

    return storage.get_notes(subj)


@app.post("/notes/{subject}", response_model=NoteResponse, tags=["Notes"])
async def create_note(subject: str, request: NoteRequest):
    """Create a new note for a specific subject."""
    try:
        subj = Subject(subject.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid subject: {subject}")

    note = storage.add_note(subj, request.content, request.tags)
    return NoteResponse(note=note, message="Note created successfully")


@app.delete("/notes/{subject}/{note_id}", tags=["Notes"])
async def delete_note(subject: str, note_id: str):
    """Delete a specific note."""
    try:
        subj = Subject(subject.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid subject: {subject}")

    if storage.delete_note(subj, note_id):
        return {"message": "Note deleted successfully"}
    raise HTTPException(status_code=404, detail="Note not found")


@app.get("/notes/{subject}/search", response_model=List[Note], tags=["Notes"])
async def search_notes(subject: str, query: str):
    """Search notes by keyword within a subject."""
    try:
        subj = Subject(subject.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid subject: {subject}")

    return storage.search_notes(subj, query)


# ========== Solutions Endpoints ==========

@app.get("/solutions/{subject}", response_model=List[Solution], tags=["Solutions"])
async def get_solutions(subject: str):
    """Get all solutions for a specific subject."""
    try:
        subj = Subject(subject.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid subject: {subject}")

    return storage.get_solutions(subj)


@app.post("/solutions/{subject}", response_model=SolutionResponse, tags=["Solutions"])
async def create_solution(subject: str, request: SolutionRequest):
    """Create a new problem-solution pair for a specific subject."""
    try:
        subj = Subject(subject.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid subject: {subject}")

    solution = storage.add_solution(subj, request.problem, request.solution, request.tags)
    return SolutionResponse(solution=solution, message="Solution saved successfully")


@app.delete("/solutions/{subject}/{solution_id}", tags=["Solutions"])
async def delete_solution(subject: str, solution_id: str):
    """Delete a specific solution."""
    try:
        subj = Subject(subject.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid subject: {subject}")

    if storage.delete_solution(subj, solution_id):
        return {"message": "Solution deleted successfully"}
    raise HTTPException(status_code=404, detail="Solution not found")


@app.get("/solutions/{subject}/search", response_model=List[Solution], tags=["Solutions"])
async def search_solutions(subject: str, query: str):
    """Search solutions by keyword within a subject."""
    try:
        subj = Subject(subject.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid subject: {subject}")

    return storage.search_solutions(subj, query)


# ========== History Endpoints ==========

@app.get("/history/{subject}", response_model=List[Message], tags=["History"])
async def get_history(subject: str, session_id: str = "default", limit: int = 20):
    """Get conversation history for a specific subject."""
    try:
        subj = Subject(subject.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid subject: {subject}")

    return storage.get_history(session_id, subj, limit)


@app.delete("/history/{subject}", tags=["History"])
async def clear_history(subject: str, session_id: str = "default"):
    """Clear conversation history for a specific subject."""
    try:
        subj = Subject(subject.lower())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid subject: {subject}")

    storage.clear_history(session_id, subj)
    return {"message": f"History cleared for {subject}"}


@app.get("/history", response_model=List[Message], tags=["History"])
async def get_all_history(session_id: str = "default", limit: int = 50):
    """Get all conversation history across all subjects."""
    return storage.get_all_history(session_id, limit)


# ========== Health Check ==========

@app.get("/health", tags=["System"])
async def health_check():
    """Check if the API is running."""
    return {
        "status": "healthy",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "tavily_configured": bool(os.getenv("TAVILY_API_KEY"))
    }


# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
