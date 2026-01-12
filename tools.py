import os
from typing import List, Optional
from langchain_core.tools import tool
from tavily import TavilyClient

from models import Subject
from storage import storage


# Initialize Tavily client for web search
tavily_client = None


def get_tavily_client():
    """Lazy initialization of Tavily client."""
    global tavily_client
    if tavily_client is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if api_key:
            tavily_client = TavilyClient(api_key=api_key)
    return tavily_client


# ========== Note Tools ==========

@tool
def save_note(content: str, subject: str, tags: Optional[List[str]] = None) -> str:
    """
    Save a study note for a specific subject.

    Args:
        content: The note content to save
        subject: The subject this note belongs to (python, langgraph, langchain, javascript, llm, automation, n8n, gohighlevel)
        tags: Optional list of tags for categorization

    Returns:
        Confirmation message with note ID
    """
    try:
        subj = Subject(subject.lower())
    except ValueError:
        subj = Subject.GENERAL

    note = storage.add_note(subj, content, tags or [])
    return f"Note saved successfully! ID: {note.id}"


@tool
def get_notes(subject: str) -> str:
    """
    Retrieve all notes for a specific subject.

    Args:
        subject: The subject to get notes for (python, langgraph, langchain, javascript, llm, automation, n8n, gohighlevel)

    Returns:
        List of notes or message if no notes found
    """
    try:
        subj = Subject(subject.lower())
    except ValueError:
        subj = Subject.GENERAL

    notes = storage.get_notes(subj)
    if not notes:
        return f"No notes found for {subject}."

    result = f"Notes for {subject} ({len(notes)} total):\n\n"
    for i, note in enumerate(notes, 1):
        tags_str = f" [Tags: {', '.join(note.tags)}]" if note.tags else ""
        result += f"{i}. {note.content}{tags_str}\n\n"

    return result


@tool
def search_notes(query: str, subject: str) -> str:
    """
    Search notes by keyword within a subject.

    Args:
        query: The search keyword
        subject: The subject to search in

    Returns:
        Matching notes or message if none found
    """
    try:
        subj = Subject(subject.lower())
    except ValueError:
        subj = Subject.GENERAL

    notes = storage.search_notes(subj, query)
    if not notes:
        return f"No notes matching '{query}' found in {subject}."

    result = f"Found {len(notes)} notes matching '{query}':\n\n"
    for i, note in enumerate(notes, 1):
        result += f"{i}. {note.content}\n\n"

    return result


# ========== Solution Tools ==========

@tool
def save_solution(problem: str, solution: str, subject: str, tags: Optional[List[str]] = None) -> str:
    """
    Save a problem-solution pair from past experience.

    Args:
        problem: Description of the problem encountered
        solution: How the problem was solved
        subject: The subject this solution belongs to
        tags: Optional list of tags for categorization

    Returns:
        Confirmation message with solution ID
    """
    try:
        subj = Subject(subject.lower())
    except ValueError:
        subj = Subject.GENERAL

    sol = storage.add_solution(subj, problem, solution, tags or [])
    return f"Solution saved successfully! ID: {sol.id}"


@tool
def get_solutions(subject: str) -> str:
    """
    Retrieve all solutions for a specific subject.

    Args:
        subject: The subject to get solutions for

    Returns:
        List of solutions or message if none found
    """
    try:
        subj = Subject(subject.lower())
    except ValueError:
        subj = Subject.GENERAL

    solutions = storage.get_solutions(subj)
    if not solutions:
        return f"No solutions found for {subject}."

    result = f"Solutions for {subject} ({len(solutions)} total):\n\n"
    for i, sol in enumerate(solutions, 1):
        result += f"{i}. Problem: {sol.problem}\n   Solution: {sol.solution}\n\n"

    return result


@tool
def search_solutions(query: str, subject: str) -> str:
    """
    Search for past solutions by keyword.

    Args:
        query: The search keyword (e.g., "async error", "import issue")
        subject: The subject to search in

    Returns:
        Matching solutions or message if none found
    """
    try:
        subj = Subject(subject.lower())
    except ValueError:
        subj = Subject.GENERAL

    solutions = storage.search_solutions(subj, query)
    if not solutions:
        return f"No solutions matching '{query}' found in {subject}."

    result = f"Found {len(solutions)} solutions matching '{query}':\n\n"
    for i, sol in enumerate(solutions, 1):
        result += f"{i}. Problem: {sol.problem}\n   Solution: {sol.solution}\n\n"

    return result


# ========== Web Search Tool ==========

@tool
def web_search(query: str) -> str:
    """
    Search the web for the latest information on a topic.

    Args:
        query: The search query (e.g., "LangGraph latest features 2024")

    Returns:
        Search results with titles, snippets, and URLs
    """
    client = get_tavily_client()
    if not client:
        return "Web search is unavailable. Please set the TAVILY_API_KEY environment variable."

    try:
        response = client.search(query=query, max_results=5)
        results = response.get("results", [])

        if not results:
            return f"No results found for '{query}'."

        result_text = f"Web search results for '{query}':\n\n"
        for i, item in enumerate(results, 1):
            title = item.get("title", "No title")
            snippet = item.get("content", "No description")
            url = item.get("url", "")
            result_text += f"{i}. {title}\n   {snippet}\n   URL: {url}\n\n"

        return result_text

    except Exception as e:
        return f"Web search failed: {str(e)}"


# List of all tools for easy import
all_tools = [
    save_note,
    get_notes,
    search_notes,
    save_solution,
    get_solutions,
    search_solutions,
    web_search,
]
