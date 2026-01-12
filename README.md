# Study Platform with Subject-Specific Chatbots

A LangGraph-powered study platform with personal assistant chatbots for different subjects. Uses automatic routing to detect which subject you're asking about.

## Subjects Supported
- Python
- LangGraph
- LangChain
- JavaScript
- LLM (Large Language Models)
- Automation
- n8n
- GoHighLevel

## How It Works

### Message Flow

```
You send: "How do I create a graph in LangGraph?"
                    |
                    v
           +---------------+
           |    ROUTER     |  GPT-4 analyzes message
           |               |  Detects: "LangGraph"
           +-------+-------+
                   |
                   v
        +---------------------+
        | LANGGRAPH ASSISTANT |  Subject-specific expert
        |                     |
        |  Has access to:     |
        |  - Your notes       |
        |  - Past solutions   |
        |  - Web search       |
        +----------+----------+
                   |
                   v
              Response with
              code examples
```

### Component Overview

| Component | Purpose |
|-----------|---------|
| **Router** | Reads your message, determines subject (Python? LangGraph? n8n?) |
| **Subject Assistant** | Expert chatbot for that subject, knows your history |
| **Notes Storage** | Save study notes like "LangGraph uses StateGraph class" |
| **Solutions Memory** | Remember "I fixed the async error by using await" |
| **Web Search** | Find latest docs, tutorials, solutions online |

### Example Conversation

```
You: "How do I connect nodes in LangGraph?"
Bot: [Router detects: LangGraph]
     "Use add_edge() method: graph.add_edge('node1', 'node2')..."

You: "Save this as a note"
Bot: [Saves to LangGraph notes]
     "Saved note about connecting nodes!"

You: "What's new in n8n version 1.0?"
Bot: [Router detects: n8n -> switches assistant]
     [Searches web for latest n8n info]
     "n8n 1.0 introduced..."

You: "How did I solve that Python async error before?"
Bot: [Router detects: Python]
     [Searches your solutions memory]
     "You solved it by using asyncio.run()..."
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Send message, get response |
| `/subjects` | GET | List all 8 subjects |
| `/notes/{subject}` | GET | Get all notes for a subject |
| `/notes/{subject}` | POST | Add a new note |
| `/solutions/{subject}` | GET | View past solutions |
| `/history/{subject}` | GET | Get conversation history |
| `/history/{subject}` | DELETE | Clear conversation history |

## Quick Start

### 1. Install Dependencies

```bash
pip install langgraph langchain-openai tavily-python fastapi uvicorn pydantic
```

### 2. Set Environment Variables

```bash
# Windows
set OPENAI_API_KEY=your-openai-key
set TAVILY_API_KEY=your-tavily-key

# Linux/Mac
export OPENAI_API_KEY=your-openai-key
export TAVILY_API_KEY=your-tavily-key
```

### 3. Run the Server

```bash
cd study_platform
uvicorn main:app --reload
```

### 4. Open Swagger UI

Navigate to http://localhost:8000/docs to test the API endpoints.

## Project Structure

```
study_platform/
  __init__.py      # Package init
  models.py        # Pydantic data models
  storage.py       # In-memory storage for notes/solutions
  tools.py         # LangGraph tools (notes, solutions, web search)
  state.py         # LangGraph state definitions
  agents.py        # Router and subject assistant agents
  graph.py         # LangGraph workflow
  main.py          # FastAPI application
```

## Tech Stack

- **LangGraph**: Multi-agent orchestration
- **OpenAI GPT-4**: LLM for all assistants
- **Tavily**: Web search tool
- **FastAPI**: REST API backend
- **Pydantic**: Data validation
- **In-memory storage**: Notes, solutions, conversation history
