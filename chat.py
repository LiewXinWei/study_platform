#!/usr/bin/env python
"""
Terminal Chatbot for Study Platform
Run: python chat.py
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from graph import chat
from models import Subject

# Colors for terminal output
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_welcome():
    print(f"""
{Colors.CYAN}{'='*60}
   STUDY PLATFORM - Your Personal Learning Assistant
{'='*60}{Colors.RESET}

Subjects: Python, LangGraph, LangChain, JavaScript, LLM,
          Automation, n8n, GoHighLevel

Commands:
  {Colors.YELLOW}/subjects{Colors.RESET}  - List all subjects
  {Colors.YELLOW}/notes{Colors.RESET}     - Show notes for current subject
  {Colors.YELLOW}/clear{Colors.RESET}     - Clear screen
  {Colors.YELLOW}/quit{Colors.RESET}      - Exit chatbot

Just type your question and I'll route it to the right assistant!
{Colors.CYAN}{'='*60}{Colors.RESET}
""")

def print_response(response: str, subject: str):
    """Print the bot's response with formatting."""
    subject_colors = {
        "python": Colors.BLUE,
        "langgraph": Colors.GREEN,
        "langchain": Colors.GREEN,
        "javascript": Colors.YELLOW,
        "llm": Colors.CYAN,
        "automation": Colors.YELLOW,
        "n8n": Colors.CYAN,
        "gohighlevel": Colors.CYAN,
        "general": Colors.RESET
    }
    color = subject_colors.get(subject, Colors.RESET)

    print(f"\n{color}[{subject.upper()} Assistant]{Colors.RESET}")
    print(f"{response}\n")

def handle_command(command: str) -> bool:
    """Handle special commands. Returns True if should continue, False to exit."""
    cmd = command.lower().strip()

    if cmd == "/quit" or cmd == "/exit" or cmd == "/q":
        print(f"\n{Colors.CYAN}Goodbye! Happy studying!{Colors.RESET}\n")
        return False

    elif cmd == "/subjects":
        subjects = [s.value for s in Subject if s != Subject.GENERAL]
        print(f"\n{Colors.GREEN}Available subjects:{Colors.RESET}")
        for s in subjects:
            print(f"  - {s}")
        print()
        return True

    elif cmd == "/clear" or cmd == "/cls":
        os.system('cls' if os.name == 'nt' else 'clear')
        print_welcome()
        return True

    elif cmd == "/notes":
        print(f"\n{Colors.YELLOW}To see notes, ask: 'Show my notes for Python'{Colors.RESET}\n")
        return True

    elif cmd == "/help":
        print(f"""
{Colors.CYAN}Commands:{Colors.RESET}
  /subjects  - List all subjects
  /notes     - How to view notes
  /clear     - Clear screen
  /quit      - Exit chatbot

{Colors.CYAN}Tips:{Colors.RESET}
  - Just ask any question, I'll detect the subject
  - Say "save this as a note" to save information
  - Ask "how did I solve..." to search past solutions
  - Ask "search the web for..." for latest info
""")
        return True

    return True

def main():
    """Main chatbot loop."""
    print_welcome()

    session_id = "terminal_session"

    while True:
        try:
            # Get user input
            user_input = input(f"{Colors.BOLD}You:{Colors.RESET} ").strip()

            if not user_input:
                continue

            # Check for commands
            if user_input.startswith("/"):
                if not handle_command(user_input):
                    break
                continue

            # Send to chat
            print(f"{Colors.YELLOW}Thinking...{Colors.RESET}", end="\r")

            result = chat(user_input, session_id)

            # Clear "Thinking..." and print response
            print(" " * 20, end="\r")
            print_response(
                result["response"],
                result["detected_subject"].value if result["detected_subject"] else "general"
            )

        except KeyboardInterrupt:
            print(f"\n\n{Colors.CYAN}Goodbye! Happy studying!{Colors.RESET}\n")
            break
        except Exception as e:
            print(f"\n{Colors.YELLOW}Error: {e}{Colors.RESET}\n")
            print("Make sure your API keys are set correctly.\n")

if __name__ == "__main__":
    main()
