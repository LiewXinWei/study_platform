import os
import re
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


# Verbosity detection keywords
VERBOSE_KEYWORDS = [
    "detail", "detailed", "deep", "deeper", "expand", "explain more",
    "step by step", "step-by-step", "elaborate", "in depth", "in-depth",
    "comprehensive", "thorough", "full explanation", "more info", "more information"
]

# SIMPLIFY intent keywords - user wants simpler explanation, NOT a topic change
SIMPLIFY_KEYWORDS = [
    "im not a tech person", "i'm not a tech person", "not a tech person",
    "i dont understand", "i don't understand", "don't understand", "dont understand",
    "confusing", "confused", "explain simpler", "explain it simpler",
    "explain like im 5", "explain like i'm 5", "explain like im 12", "explain like i'm 12",
    "eli5", "in order", "in order way", "tell me in order",
    "break it down", "break down", "use simple words", "simpler words",
    "too technical", "too complicated", "too complex", "what does that mean",
    "what do you mean", "explain that", "can you simplify", "simplify",
    "dumb it down", "layman terms", "layman's terms", "plain english",
    "beginner friendly", "for beginners", "simple explanation"
]


def detect_verbosity(message: str) -> bool:
    """Check if user is requesting detailed/verbose response."""
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in VERBOSE_KEYWORDS)


def detect_simplify_intent(message: str) -> bool:
    """Check if user is asking for a simpler explanation (NOT a topic change)."""
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in SIMPLIFY_KEYWORDS)


def detect_code_request(message: str) -> bool:
    """Check if user explicitly asks for code."""
    code_keywords = [
        "show code", "code example", "example code", "implementation",
        "snippet", "write code", "give me code", "show me code",
        "how to code", "code it", "sample code", "code sample"
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in code_keywords)


# Base concise response rules (prepended to all prompts)
CONCISE_RULES = """
RESPONSE STYLE RULES (ALWAYS FOLLOW):
- Keep responses SHORT by default: max 2-3 sentences OR max 3 bullet points
- Use plain English first, avoid jargon unless necessary
- NO code blocks unless the user explicitly asks for code (e.g., "show code", "example", "implementation", "snippet")
- If user asks for more detail (e.g., "explain more", "go deeper", "step by step"), then expand
- Focus on the key insight, not exhaustive coverage

"""

# Character limit for auto-rewrite
MAX_RESPONSE_LENGTH = 1200

# Simple mode prompt - for beginner-friendly, ordered explanations
SIMPLE_MODE_PROMPT = """
[SIMPLE MODE - BEGINNER FRIENDLY EXPLANATION]
The user has indicated they need a simpler explanation. Follow this EXACT format:

1. **What it is (1 sentence):** Give a simple, jargon-free definition.

2. **3 Key Ideas:**
   - First key point in plain language
   - Second key point in plain language
   - Third key point in plain language

3. **Everyday Analogy:** Compare it to something from daily life (cooking, driving, organizing, etc.)

4. **Mini Example:** (Keep it very simple, no complex code)

5. **Check Question:** Ask ONE simple yes/no or A/B question to confirm understanding.
   Example: "Does that make sense?" or "Which sounds clearer: A or B?"

IMPORTANT RULES FOR SIMPLE MODE:
- NO jargon. If you must use a technical term, define it immediately in parentheses.
- Use short sentences. One idea per sentence.
- Speak like you're explaining to a smart 12-year-old.
- Be encouraging, not condescending.

"""


def rewrite_to_concise(response: str, llm) -> str:
    """Rewrite a long response to a concise version using LLM."""
    rewrite_prompt = """Rewrite the following response into a CONCISE version:
- Max 3 bullet points
- Keep the same meaning
- Plain English, no jargon
- Remove any code blocks
- Focus on the key takeaway

Original response:
{response}

Concise version:"""

    result = llm.invoke([
        SystemMessage(content=rewrite_prompt.format(response=response))
    ])
    return result.content


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

    Key logic:
    - If SIMPLIFY intent detected AND we have a current_subject, keep that subject
      and switch style to "simple" (do NOT route to GENERAL)
    - Otherwise, classify normally and set style to "normal"
    """
    user_message = state.get("user_message", "")
    current_subject = state.get("current_subject")
    last_question = state.get("last_subject_question")

    # Check for SIMPLIFY intent first
    is_simplify = detect_simplify_intent(user_message)

    # If user wants simplification AND we have a current subject context, keep it
    if is_simplify and current_subject is not None:
        return {
            "current_subject": current_subject,
            "style": "simple",
            "user_level": "beginner"
        }

    # Normal routing: classify the message to determine subject
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
        SystemMessage(content=router_prompt.format(message=user_message))
    ])

    # Parse the response to get the subject
    detected = response.content.strip().lower()

    try:
        subject = Subject(detected)
    except ValueError:
        subject = Subject.GENERAL

    # Determine if this is a meaningful subject question (to store for later "explain that" follow-ups)
    # A meaningful question is one that's routed to a specific subject (not GENERAL) and not a simplify request
    new_last_question = last_question  # default: keep existing
    if subject != Subject.GENERAL and not is_simplify:
        new_last_question = user_message

    return {
        "current_subject": subject,
        "style": "normal",
        "last_subject_question": new_last_question
    }


def assistant_node(state: StudyState) -> dict:
    """
    Main assistant node that responds based on the detected subject.
    Uses tools for notes, solutions, and web search.
    Applies concise response rules by default.

    When style="simple", uses beginner-friendly ordered explanation format.
    """
    llm = get_llm()
    subject = state.get("current_subject", Subject.GENERAL)
    user_message = state.get("user_message", "")
    style = state.get("style", "normal")
    last_question = state.get("last_subject_question", "")

    # Detect if user wants verbose/detailed response
    verbose_mode = detect_verbosity(user_message)
    code_requested = detect_code_request(user_message)

    # Check if simple mode is active
    simple_mode = (style == "simple")

    # Build system prompt
    if simple_mode:
        # Use simple mode prompt for beginner-friendly explanations
        system_prompt = SIMPLE_MODE_PROMPT
        # If user said "explain that" or similar and we have a last question, reference it
        if last_question and detect_simplify_intent(user_message):
            system_prompt += f"\nThe user is asking for a simpler explanation of their previous question: \"{last_question}\"\n"
    else:
        # Normal mode: use concise rules
        system_prompt = CONCISE_RULES

    # Add verbose mode override if detected (not in simple mode, as simple mode has its own format)
    if verbose_mode and not simple_mode:
        system_prompt += "\n[VERBOSE MODE]: User requested detailed explanation. You may provide longer, more comprehensive responses.\n"

    # Add code permission if detected
    if code_requested and not simple_mode:
        system_prompt += "\n[CODE REQUESTED]: User explicitly asked for code. You may include code examples.\n"

    # Add subject-specific prompt
    system_prompt += SUBJECT_PROMPTS.get(subject, SUBJECT_PROMPTS[Subject.GENERAL])

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

    # Store verbose_mode flag for post-processing
    return {"messages": [response], "verbose_mode": verbose_mode}


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
