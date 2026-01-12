from typing import Dict, List, Optional
from datetime import datetime
import uuid

from models import Subject, Note, Solution, Message


class StudyStorage:
    """In-memory storage for study platform data."""

    def __init__(self):
        # Notes organized by subject
        self._notes: Dict[Subject, List[Note]] = {subject: [] for subject in Subject}

        # Solutions organized by subject
        self._solutions: Dict[Subject, List[Solution]] = {subject: [] for subject in Subject}

        # Conversation history by session_id, then by subject
        self._history: Dict[str, Dict[Subject, List[Message]]] = {}

    # ========== Notes Methods ==========

    def add_note(self, subject: Subject, content: str, tags: List[str] = None) -> Note:
        """Add a new note for a subject."""
        note = Note(
            id=str(uuid.uuid4()),
            subject=subject,
            content=content,
            tags=tags or [],
            created_at=datetime.now()
        )
        self._notes[subject].append(note)
        return note

    def get_notes(self, subject: Subject) -> List[Note]:
        """Get all notes for a subject."""
        return self._notes[subject]

    def search_notes(self, subject: Subject, query: str) -> List[Note]:
        """Search notes by content or tags."""
        query_lower = query.lower()
        results = []
        for note in self._notes[subject]:
            if query_lower in note.content.lower():
                results.append(note)
            elif any(query_lower in tag.lower() for tag in note.tags):
                results.append(note)
        return results

    def delete_note(self, subject: Subject, note_id: str) -> bool:
        """Delete a note by ID."""
        for i, note in enumerate(self._notes[subject]):
            if note.id == note_id:
                self._notes[subject].pop(i)
                return True
        return False

    # ========== Solutions Methods ==========

    def add_solution(self, subject: Subject, problem: str, solution: str, tags: List[str] = None) -> Solution:
        """Add a new problem-solution pair."""
        sol = Solution(
            id=str(uuid.uuid4()),
            subject=subject,
            problem=problem,
            solution=solution,
            tags=tags or [],
            created_at=datetime.now()
        )
        self._solutions[subject].append(sol)
        return sol

    def get_solutions(self, subject: Subject) -> List[Solution]:
        """Get all solutions for a subject."""
        return self._solutions[subject]

    def search_solutions(self, subject: Subject, query: str) -> List[Solution]:
        """Search solutions by problem description or tags."""
        query_lower = query.lower()
        results = []
        for sol in self._solutions[subject]:
            if query_lower in sol.problem.lower():
                results.append(sol)
            elif query_lower in sol.solution.lower():
                results.append(sol)
            elif any(query_lower in tag.lower() for tag in sol.tags):
                results.append(sol)
        return results

    def delete_solution(self, subject: Subject, solution_id: str) -> bool:
        """Delete a solution by ID."""
        for i, sol in enumerate(self._solutions[subject]):
            if sol.id == solution_id:
                self._solutions[subject].pop(i)
                return True
        return False

    # ========== History Methods ==========

    def add_message(self, session_id: str, subject: Subject, role: str, content: str) -> Message:
        """Add a message to conversation history."""
        if session_id not in self._history:
            self._history[session_id] = {subj: [] for subj in Subject}

        message = Message(
            role=role,
            content=content,
            subject=subject,
            timestamp=datetime.now()
        )
        self._history[session_id][subject].append(message)
        return message

    def get_history(self, session_id: str, subject: Subject, limit: int = 20) -> List[Message]:
        """Get conversation history for a session and subject."""
        if session_id not in self._history:
            return []
        return self._history[session_id][subject][-limit:]

    def clear_history(self, session_id: str, subject: Optional[Subject] = None) -> bool:
        """Clear conversation history."""
        if session_id not in self._history:
            return False

        if subject:
            self._history[session_id][subject] = []
        else:
            self._history[session_id] = {subj: [] for subj in Subject}
        return True

    def get_all_history(self, session_id: str, limit: int = 50) -> List[Message]:
        """Get all conversation history across subjects for a session."""
        if session_id not in self._history:
            return []

        all_messages = []
        for subject_messages in self._history[session_id].values():
            all_messages.extend(subject_messages)

        # Sort by timestamp and return most recent
        all_messages.sort(key=lambda m: m.timestamp)
        return all_messages[-limit:]


# Global storage instance
storage = StudyStorage()
