import os
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from models import Subject
from state import StudyState
from tools import all_tools


# Initialize the LLM
def get_llm():
    """Get the OpenAI LLM instance."""
    return ChatOpenAI(
        model="gpt-4o-mini",  # Use gpt-4 for better results, gpt-4o-mini for speed/cost
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )


# Subject-specific system prompts
SUBJECT_PROMPTS = {
    Subject.PYTHON: """You are an expert Python programming assistant. You help with:
- Python syntax, best practices, and idioms
- Libraries like pandas, numpy, requests, etc.
- Debugging and error handling
- Code optimization and refactoring
- Testing with pytest, unittest
Always provide clear code examples and explanations.""",

    Subject.LANGGRAPH: """You are an expert LangGraph assistant. You help with:
- Building multi-agent systems with LangGraph
- StateGraph, MessageGraph, and workflow design
- Nodes, edges, and conditional routing
- Tool integration and agent orchestration
- Memory and persistence patterns
Always reference the latest LangGraph patterns and best practices.""",

    Subject.LANGCHAIN: """You are an expert LangChain assistant. You help with:
- Building LLM-powered applications
- Chains, agents, and tools
- Document loaders and text splitters
- Vector stores and retrievers
- Prompt templates and output parsers
Always provide working code examples with proper imports.""",

    Subject.JAVASCRIPT: """You are an expert JavaScript/TypeScript assistant. You help with:
- Modern ES6+ JavaScript features
- Node.js and npm ecosystem
- Frontend frameworks (React, Vue, etc.)
- Async/await and Promises
- Testing and debugging
Always provide clear, modern JavaScript code examples.""",

    Subject.LLM: """You are an expert in Large Language Models (LLMs). You help with:
- Understanding LLM architectures (GPT, Claude, etc.)
- Prompt engineering techniques
- Fine-tuning and RAG patterns
- Token optimization and cost management
- API integration best practices
Always explain concepts clearly with practical examples.""",

    Subject.AUTOMATION: """You are an expert in automation and workflow design. You help with:
- Process automation strategies
- API integrations and webhooks
- Scheduled tasks and triggers
- Error handling in automated workflows
- Best practices for reliable automation
Always provide practical, production-ready solutions.""",

    Subject.N8N: """You are an expert n8n workflow automation assistant. You help with:
- Building n8n workflows and nodes
- Custom nodes and credentials
- Data transformation and mapping
- Webhook and trigger configurations
- Error handling and workflow debugging
Always provide specific n8n node configurations and examples.""",

    Subject.GOHIGHLEVEL: """You are an expert GoHighLevel (GHL) assistant. You help with:
- CRM setup and customization
- Workflow automations in GHL
- API integrations and webhooks
- Funnel and landing page building
- SMS and email campaign setup
Always provide actionable GHL-specific guidance.""",

    Subject.GENERAL: """You are a helpful study assistant. You can help with various topics.
If you're unsure about the subject, ask for clarification.
You have access to tools for saving notes, recalling past solutions, and searching the web."""
}


def router_node(state: StudyState) -> dict:
    """
    Router node that classifies the user's message to determine the appropriate subject.
    """
    llm = get_llm()

    router_prompt = """You are a message classifier. Analyze the user's message and determine which subject it belongs to.

Available subjects:
- python: Python programming language questions
- langgraph: LangGraph multi-agent framework questions
- langchain: LangChain LLM framework questions
- javascript: JavaScript/TypeScript questions
- llm: Large Language Model concepts and usage
- automation: General automation and workflow questions
- n8n: n8n workflow automation platform questions
- gohighlevel: GoHighLevel CRM platform questions
- general: If the message doesn't fit any specific subject

Respond with ONLY the subject name in lowercase, nothing else.

User message: {message}"""

    response = llm.invoke([
        SystemMessage(content=router_prompt.format(message=state["user_message"]))
    ])

    # Parse the response to get the subject
    detected = response.content.strip().lower()

    try:
        subject = Subject(detected)
    except ValueError:
        subject = Subject.GENERAL

    return {"current_subject": subject}


def assistant_node(state: StudyState) -> dict:
    """
    Main assistant node that responds based on the detected subject.
    Uses tools for notes, solutions, and web search.
    """
    llm = get_llm()
    subject = state.get("current_subject", Subject.GENERAL)

    # Get the appropriate system prompt
    system_prompt = SUBJECT_PROMPTS.get(subject, SUBJECT_PROMPTS[Subject.GENERAL])

    # Add tool usage instructions
    system_prompt += """

You have access to these tools:
- save_note: Save a study note for future reference
- get_notes: Retrieve saved notes for a subject
- search_notes: Search notes by keyword
- save_solution: Save a problem-solution pair from past experience
- get_solutions: Get all saved solutions
- search_solutions: Search for past solutions by keyword
- web_search: Search the web for the latest information

When the user asks you to:
- "Save this as a note" or "Remember this" → Use save_note
- "What notes do I have?" or "Show my notes" → Use get_notes
- "How did I solve..." or "What was my solution for..." → Use search_solutions
- "Search for..." or "Find the latest..." → Use web_search

Always be helpful and provide clear, actionable responses. The current subject is: """ + subject.value

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(all_tools)

    # Create messages list
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])

    # Get response
    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}


def tool_node(state: StudyState) -> dict:
    """
    Execute tools called by the assistant.
    """
    from langgraph.prebuilt import ToolNode

    tool_executor = ToolNode(all_tools)
    return tool_executor.invoke(state)


def should_continue(state: StudyState) -> Literal["tools", "end"]:
    """
    Determine if we should continue to tool execution or end.
    """
    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM made a tool call, continue to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Otherwise, end the conversation turn
    return "end"
