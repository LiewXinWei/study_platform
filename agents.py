import os
import re
import json
from typing import Literal, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from models import Subject
from state import StudyState
from tools import all_tools


# JSON Router Schema for structured routing decisions
ROUTER_JSON_SCHEMA = {
    "mode": "ANSWER|TEACH|QUIZ|DEBUG|ASK_CLARIFY|SIMPLIFY",
    "topic": "LangGraph|Python|LangChain|JavaScript|LLM|Automation|n8n|GoHighLevel|Unknown",
    "confidence": "number 0-1",
    "missing_info": "list of strings",
    "reason": "short explanation"
}


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
    "what do you mean", "what do u mean", "explain that", "can you simplify", "simplify",
    "dumb it down", "layman terms", "layman's terms", "plain english",
    "beginner friendly", "for beginners", "simple explanation",
    "what do u mean by that", "what does that mean by that", "huh", "what"
]


def parse_router_json(response_text: str) -> Optional[dict]:
    """
    Parse JSON from router response. Handles markdown code blocks and raw JSON.
    Returns None if parsing fails (triggers fallback to ASK_CLARIFY).
    """
    # Try to extract JSON from markdown code block
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw JSON
    try:
        # Find JSON object in response
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start != -1 and end > start:
            return json.loads(response_text[start:end])
    except json.JSONDecodeError:
        pass

    return None


def validate_router_result(result: dict) -> dict:
    """
    Validate and normalize router result. Fill in defaults for missing fields.
    """
    valid_modes = ["ANSWER", "TEACH", "QUIZ", "DEBUG", "ASK_CLARIFY", "SIMPLIFY"]
    valid_topics = ["LangGraph", "Python", "LangChain", "JavaScript", "LLM",
                    "Automation", "n8n", "GoHighLevel", "Unknown"]

    mode = result.get("mode", "ASK_CLARIFY").upper()
    if mode not in valid_modes:
        mode = "ASK_CLARIFY"

    topic = result.get("topic", "Unknown")
    # Normalize topic capitalization
    topic_lower = topic.lower()
    topic_map = {t.lower(): t for t in valid_topics}
    topic = topic_map.get(topic_lower, "Unknown")

    confidence = result.get("confidence", 0.5)
    if not isinstance(confidence, (int, float)):
        confidence = 0.5
    confidence = max(0.0, min(1.0, float(confidence)))

    missing_info = result.get("missing_info", [])
    if not isinstance(missing_info, list):
        missing_info = []

    reason = result.get("reason", "")

    return {
        "mode": mode,
        "topic": topic,
        "confidence": confidence,
        "missing_info": missing_info,
        "reason": reason
    }


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
The user has indicated they need a simpler explanation. Follow this EXACT format IN ORDER:

1. **What it is (1 sentence):** Give a simple, jargon-free definition.

2. **3 Key Ideas:**
   - First key point in plain language
   - Second key point in plain language
   - Third key point in plain language

3. **Everyday Analogy:** Compare it to something from daily life (cooking, driving, organizing, etc.)

4. **Mini Example:** A tiny pseudo-code or flow diagram, NOT complex code. Example:
   ```
   User asks question → Router picks topic → Assistant answers → Done
   ```

5. **Check Question:** Ask ONE simple yes/no or A/B question to confirm understanding.
   Example: "Does that make sense?" or "Which sounds clearer: A or B?"

IMPORTANT RULES FOR SIMPLE MODE:
- NO jargon. If you must use a technical term, define it immediately in parentheses.
- Use short sentences. One idea per sentence.
- Speak like you're explaining to a smart 12-year-old.
- Be encouraging, not condescending.
- For LangGraph topics, always connect to real mechanisms: state, nodes, edges, reducers.

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

    Subject.LANGGRAPH: """You are an expert LangGraph assistant. You MUST be specific and accurate about LangGraph mechanisms.

CORE LANGGRAPH CONCEPTS YOU MUST KNOW:
1. **State & Reducers**: State is a TypedDict. Without reducers, LangGraph uses "last-write-wins" - if two nodes write to the same key, one overwrites the other. Use `Annotated[list, add_messages]` to APPEND instead.

2. **Nodes & Edges**: Nodes are functions that take state and return partial state updates. Edges connect nodes. Conditional edges use a function to pick the next node.

3. **Checkpointing**: Enables persistence and "time travel". Use MemorySaver or SqliteSaver. Allows interrupt/resume of graph execution.

4. **Interrupts**: Use `interrupt_before` or `interrupt_after` on nodes to pause execution for human-in-the-loop. Resume with `graph.invoke(None, config)`.

5. **Tool Routing**: Bind tools to LLM with `llm.bind_tools()`. Check `tool_calls` on AIMessage to route to tool execution.

RESPONSE RULES:
- When style="normal": Provide technical depth with at least 1 concrete code example.
- NEVER be generic. Always mention at least ONE specific LangGraph mechanism when relevant:
  * Reducers / state merge / add_messages
  * Conditional edges / should_continue pattern
  * Loop termination conditions
  * Checkpointing / MemorySaver / SqliteSaver
  * interrupt_before / interrupt_after
  * Tool calls and ToolNode routing

Example of GOOD specificity:
"Without a reducer like `add_messages`, parallel nodes writing to `messages` would cause last-write-wins - you'd lose one node's output."

Example of BAD generic answer:
"LangGraph helps you build multi-agent systems." (Too vague, no mechanism mentioned)""",

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
    Router node that classifies the user's message using structured JSON routing.

    Returns JSON schema:
    {
        "mode": "ANSWER"|"TEACH"|"QUIZ"|"DEBUG"|"ASK_CLARIFY"|"SIMPLIFY",
        "topic": "LangGraph"|"Python"|...|"Unknown",
        "confidence": 0.0-1.0,
        "missing_info": ["list of missing info"],
        "reason": "short explanation"
    }

    Key routing rules:
    1. If SIMPLIFY intent detected AND current_subject exists -> mode="SIMPLIFY", keep topic
    2. If confidence < 0.65 -> mode="ASK_CLARIFY" with 1 targeted question
    3. Only route to Unknown if user truly changes topic away from study subjects
    """
    user_message = state.get("user_message", "")
    current_subject = state.get("current_subject")
    last_question = state.get("last_subject_question")
    last_answer = state.get("last_assistant_answer", "")
    attempts_clarify = state.get("attempts_clarify", 0)

    # Check for SIMPLIFY intent first (keyword-based, fast path)
    is_simplify = detect_simplify_intent(user_message)

    # RULE 1: If user wants simplification AND we have a current subject context, keep it
    if is_simplify and current_subject is not None:
        router_result = {
            "mode": "SIMPLIFY",
            "topic": current_subject.value.capitalize() if current_subject != Subject.GENERAL else "Unknown",
            "confidence": 1.0,
            "missing_info": [],
            "reason": "User requested simpler explanation of current topic"
        }
        return {
            "current_subject": current_subject,
            "style": "simple",
            "user_level": "beginner",
            "router_result": router_result,
            "attempts_clarify": 0  # Reset on valid interaction
        }

    # Use LLM for structured JSON routing
    llm = get_llm()

    router_prompt = """You are a message router for a Study Buddy chatbot. Analyze the user's message and return a JSON decision.

AVAILABLE TOPICS:
- LangGraph: Multi-agent framework, StateGraph, nodes, edges, reducers, checkpointing
- Python: Python programming language
- LangChain: LLM application framework
- JavaScript: JavaScript/TypeScript
- LLM: Large Language Model concepts
- Automation: General automation/workflows
- n8n: n8n workflow platform
- GoHighLevel: GoHighLevel CRM
- Unknown: Only if truly unrelated to any topic above

MODES:
- ANSWER: Direct answer to a clear question
- TEACH: Explain a concept in depth
- QUIZ: User wants to test their knowledge
- DEBUG: User has an error or bug to fix
- ASK_CLARIFY: Question is ambiguous, need 1 targeted clarification
- SIMPLIFY: User wants simpler explanation (detected by phrases like "I don't understand")

RULES:
1. If the message sounds like confusion/simplification request AND there was a previous topic, keep that topic and use mode=SIMPLIFY
2. If confidence < 0.65, use mode=ASK_CLARIFY
3. Only use topic=Unknown if the message is truly unrelated to all study topics

Current context:
- Previous topic: {current_topic}
- Previous question: {last_question}

User message: {message}

Respond with ONLY valid JSON (no markdown, no explanation):
{{"mode": "...", "topic": "...", "confidence": 0.0-1.0, "missing_info": [], "reason": "..."}}"""

    context_topic = current_subject.value if current_subject else "None"

    response = llm.invoke([
        SystemMessage(content=router_prompt.format(
            message=user_message,
            current_topic=context_topic,
            last_question=last_question or "None"
        ))
    ])

    # Parse JSON response with fallback
    parsed = parse_router_json(response.content)

    if parsed is None:
        # Fallback: JSON parsing failed, use ASK_CLARIFY safely
        router_result = {
            "mode": "ASK_CLARIFY",
            "topic": context_topic if current_subject else "Unknown",
            "confidence": 0.3,
            "missing_info": ["Could not parse routing decision"],
            "reason": "JSON parsing failed, requesting clarification"
        }
    else:
        router_result = validate_router_result(parsed)

    # RULE 2: If confidence < 0.65 and not too many clarify attempts, ask for clarification
    if router_result["confidence"] < 0.65 and attempts_clarify < 2:
        router_result["mode"] = "ASK_CLARIFY"

    # Map topic to Subject enum
    topic_to_subject = {
        "langgraph": Subject.LANGGRAPH,
        "python": Subject.PYTHON,
        "langchain": Subject.LANGCHAIN,
        "javascript": Subject.JAVASCRIPT,
        "llm": Subject.LLM,
        "automation": Subject.AUTOMATION,
        "n8n": Subject.N8N,
        "gohighlevel": Subject.GOHIGHLEVEL,
        "unknown": Subject.GENERAL
    }
    topic_lower = router_result["topic"].lower()
    subject = topic_to_subject.get(topic_lower, Subject.GENERAL)

    # Determine style based on mode
    style = "simple" if router_result["mode"] == "SIMPLIFY" else "normal"

    # Update last_subject_question if this is a meaningful subject question
    new_last_question = last_question
    if subject != Subject.GENERAL and router_result["mode"] not in ["SIMPLIFY", "ASK_CLARIFY"]:
        new_last_question = user_message

    # Track clarify attempts
    new_attempts = attempts_clarify + 1 if router_result["mode"] == "ASK_CLARIFY" else 0

    return {
        "current_subject": subject,
        "style": style,
        "router_result": router_result,
        "last_subject_question": new_last_question,
        "attempts_clarify": new_attempts
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

    # Extract response content for storing as last_assistant_answer
    response_text = response.content if hasattr(response, "content") else str(response)

    # Store verbose_mode flag and last answer for post-processing and follow-ups
    return {
        "messages": [response],
        "verbose_mode": verbose_mode,
        "last_assistant_answer": response_text
    }


def tool_node(state: StudyState) -> dict:
    """
    Execute tools called by the assistant.
    """
    from langgraph.prebuilt import ToolNode

    tool_executor = ToolNode(all_tools)
    return tool_executor.invoke(state)


def should_continue(state: StudyState) -> Literal["tools", "verifier", "end"]:
    """
    Determine if we should continue to tool execution, verifier, or end.
    """
    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM made a tool call, continue to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # If verifier hasn't run yet, go to verifier
    if not state.get("verifier_passed", False):
        return "verifier"

    # Otherwise, end the conversation turn
    return "end"


def verifier_node(state: StudyState) -> dict:
    """
    Verifier node that checks the quality of the assistant's response.

    Checks:
    1. Did we answer the question directly?
    2. Are there vague claims without specific mechanisms?
    3. If asking for specifics and we don't know, do we admit uncertainty?

    If verification fails, sets verifier_passed=False and provides feedback.
    If verification passes, sets verifier_passed=True.
    """
    messages = state.get("messages", [])
    user_message = state.get("user_message", "")
    subject = state.get("current_subject", Subject.GENERAL)
    style = state.get("style", "normal")

    if not messages:
        return {"verifier_passed": True}

    last_message = messages[-1]
    response_text = last_message.content if hasattr(last_message, "content") else str(last_message)

    # Skip verification for tool calls (they'll loop back)
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return {"verifier_passed": True}

    # Skip verification for simple mode (already structured)
    if style == "simple":
        return {"verifier_passed": True}

    # Skip for very short responses (likely clarification questions)
    if len(response_text) < 100:
        return {"verifier_passed": True}

    # Only verify LangGraph responses for specificity
    if subject != Subject.LANGGRAPH:
        return {"verifier_passed": True}

    # Use LLM to verify response quality
    llm = get_llm()

    verifier_prompt = """You are a response quality verifier for a LangGraph tutoring chatbot.

USER QUESTION: {user_question}

ASSISTANT RESPONSE:
{response}

VERIFY these criteria:
1. DIRECTNESS: Does the response directly answer the user's question?
2. SPECIFICITY: Does it mention at least ONE specific LangGraph mechanism (reducers, add_messages, checkpointing, interrupts, conditional edges, ToolNode, etc.)?
3. HONESTY: If the response makes claims, are they concrete or vague?

PASS if:
- Response is direct and mentions specific mechanisms
- OR response honestly says "I'm not sure" when uncertain

FAIL if:
- Response is vague/generic with no specific LangGraph mechanisms mentioned
- Response doesn't answer the actual question
- Response makes hand-wavy claims without concrete details

Respond with ONLY valid JSON:
{{"pass": true/false, "reason": "short explanation", "suggestion": "improvement if failed"}}"""

    try:
        result = llm.invoke([
            SystemMessage(content=verifier_prompt.format(
                user_question=user_message,
                response=response_text[:1500]  # Truncate for cost
            ))
        ])

        parsed = parse_router_json(result.content)
        if parsed and not parsed.get("pass", True):
            return {
                "verifier_passed": False,
                "verifier_feedback": parsed.get("suggestion", "Be more specific about LangGraph mechanisms.")
            }
    except Exception:
        # If verification fails, let it pass to avoid blocking
        pass

    return {"verifier_passed": True}


def revise_node(state: StudyState) -> dict:
    """
    Revise node that improves the response based on verifier feedback.
    Only called when verifier_passed=False.
    """
    messages = state.get("messages", [])
    feedback = state.get("verifier_feedback", "Be more specific.")
    user_message = state.get("user_message", "")
    subject = state.get("current_subject", Subject.GENERAL)

    if not messages:
        return {"verifier_passed": True}

    last_message = messages[-1]
    original_response = last_message.content if hasattr(last_message, "content") else str(last_message)

    llm = get_llm()

    revise_prompt = """You are revising a LangGraph tutoring response to be more specific and accurate.

ORIGINAL USER QUESTION: {user_question}

ORIGINAL RESPONSE (needs improvement):
{original}

FEEDBACK: {feedback}

REVISION RULES:
1. Keep the same structure and length
2. Add at least ONE specific LangGraph mechanism (reducers, add_messages, checkpointing, interrupts, conditional edges, etc.)
3. Replace vague claims with concrete explanations
4. If you don't know something, say "I'm not sure about X, but..."

Write the REVISED response only (no meta-commentary):"""

    result = llm.invoke([
        SystemMessage(content=revise_prompt.format(
            user_question=user_message,
            original=original_response,
            feedback=feedback
        ))
    ])

    # Replace the last message with revised version
    revised_message = AIMessage(content=result.content)

    return {
        "messages": [revised_message],
        "verifier_passed": True,  # Mark as passed after revision
        "last_assistant_answer": result.content
    }


def should_revise(state: StudyState) -> Literal["revise", "end"]:
    """Determine if we need to revise the response."""
    if not state.get("verifier_passed", True):
        return "revise"
    return "end"
