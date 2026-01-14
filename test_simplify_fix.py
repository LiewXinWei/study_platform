"""
Test script to verify the SIMPLIFY intent bug fix.

This test proves that when a user says "im not a tech person" or similar
after asking a LangGraph question, the router:
1. Keeps the same subject (LangGraph) - does NOT switch to GENERAL
2. Switches style to "simple" for beginner-friendly explanations
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from graph import chat, clear_session
from models import Subject


def print_result(turn: int, user_msg: str, result: dict):
    """Pretty print a conversation turn."""
    print(f"\n{'='*60}")
    print(f"TURN {turn}")
    print(f"{'='*60}")
    print(f"USER: {user_msg}")
    print(f"\nDETECTED SUBJECT: {result['detected_subject']}")
    print(f"STYLE: {result.get('style', 'normal')}")
    print(f"\nBOT RESPONSE:")
    print("-" * 40)
    # Truncate long responses for readability
    response = result['response']
    if len(response) > 800:
        print(response[:800] + "\n... [truncated]")
    else:
        print(response)
    print("-" * 40)


def test_simplify_intent():
    """
    Test the SIMPLIFY intent detection and subject preservation.

    Expected behavior:
    - Turn 1: LangGraph question -> subject=langgraph, style=normal
    - Turn 2: "im not a tech person" -> subject=langgraph (NOT general!), style=simple
    - Turn 3: "tell me in order way" -> subject=langgraph, style=simple
    """
    print("\n" + "=" * 60)
    print("TEST: SIMPLIFY INTENT BUG FIX")
    print("=" * 60)

    # Use a unique session ID for this test
    session_id = "test_simplify_fix"

    # Clear any existing session state
    clear_session(session_id)

    # Test conversation
    test_messages = [
        "what is the main concept of LangGraph",
        "im not a tech person",
        "tell me in order way"
    ]

    results = []

    for i, msg in enumerate(test_messages, 1):
        result = chat(msg, session_id=session_id)
        results.append(result)
        print_result(i, msg, result)

    # Validate results
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    all_passed = True

    # Turn 1: Should be LangGraph, normal style
    if results[0]['detected_subject'] == Subject.LANGGRAPH:
        print("[PASS] Turn 1: Correctly detected subject as LANGGRAPH")
    else:
        print(f"[FAIL] Turn 1: Expected LANGGRAPH, got {results[0]['detected_subject']}")
        all_passed = False

    # Turn 2: Should STILL be LangGraph (not GENERAL!), simple style
    if results[1]['detected_subject'] == Subject.LANGGRAPH:
        print("[PASS] Turn 2: Subject preserved as LANGGRAPH (not switched to GENERAL)")
    else:
        print(f"[FAIL] Turn 2: Expected LANGGRAPH, got {results[1]['detected_subject']}")
        all_passed = False

    if results[1].get('style') == 'simple':
        print("[PASS] Turn 2: Style correctly set to 'simple'")
    else:
        print(f"[FAIL] Turn 2: Expected style='simple', got {results[1].get('style')}")
        all_passed = False

    # Turn 3: Should still be LangGraph, simple style
    if results[2]['detected_subject'] == Subject.LANGGRAPH:
        print("[PASS] Turn 3: Subject still LANGGRAPH")
    else:
        print(f"[FAIL] Turn 3: Expected LANGGRAPH, got {results[2]['detected_subject']}")
        all_passed = False

    if results[2].get('style') == 'simple':
        print("[PASS] Turn 3: Style still 'simple'")
    else:
        print(f"[FAIL] Turn 3: Expected style='simple', got {results[2].get('style')}")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED! The bug is fixed.")
    else:
        print("SOME TESTS FAILED. Please check the implementation.")
    print("=" * 60)

    # Cleanup
    clear_session(session_id)

    return all_passed


def test_additional_simplify_phrases():
    """Test additional simplify phrases to ensure they all work."""
    print("\n" + "=" * 60)
    print("TEST: ADDITIONAL SIMPLIFY PHRASES")
    print("=" * 60)

    session_id = "test_additional_phrases"

    simplify_phrases = [
        "i don't understand",
        "this is confusing",
        "explain like im 5",
        "break it down for me",
        "use simple words please"
    ]

    all_passed = True

    for phrase in simplify_phrases:
        # Reset session and ask a subject question first
        clear_session(session_id)

        # First turn: establish subject
        chat("how does LangGraph handle state", session_id=session_id)

        # Second turn: use simplify phrase
        result = chat(phrase, session_id=session_id)

        if result['detected_subject'] == Subject.LANGGRAPH:
            print(f"[PASS] '{phrase}' -> subject preserved as LANGGRAPH")
        else:
            print(f"[FAIL] '{phrase}' -> switched to {result['detected_subject']}")
            all_passed = False

        if result.get('style') == 'simple':
            print(f"  [PASS] style='simple'")
        else:
            print(f"  [FAIL] style='{result.get('style')}' (expected 'simple')")
            all_passed = False

    clear_session(session_id)
    return all_passed


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set it in .env file or environment.")
        exit(1)

    print("Running SIMPLIFY intent bug fix tests...")
    print("This will make actual API calls to test the fix.\n")

    # Run main test
    test1_passed = test_simplify_intent()

    # Run additional phrase tests
    test2_passed = test_additional_simplify_phrases()

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    if test1_passed and test2_passed:
        print("ALL TESTS PASSED!")
    else:
        print("Some tests failed. Review output above.")
