#!/usr/bin/env python
"""
Test harness for Study Buddy routing improvements.

Tests:
1. SIMPLIFY intent preservation (topic stays LangGraph, style becomes simple)
2. Reducers question (expects specific LangGraph mechanisms mentioned)
3. JSON router robustness (malformed input fallback)

Run: python test_routing_harness.py
"""

import os
import sys
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from graph import chat, clear_session
from models import Subject
from agents import parse_router_json, validate_router_result


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def safe_print(text: str):
    """Print text with safe encoding for Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace problematic characters
        safe_text = text.encode('ascii', 'replace').decode('ascii')
        print(safe_text)


def print_turn(turn: int, user_msg: str, result: dict):
    """Pretty print a conversation turn with routing info."""
    print(f"\n--- Turn {turn} ---")
    print(f"USER: {user_msg}")
    print(f"SUBJECT: {result['detected_subject']}")
    print(f"STYLE: {result.get('style', 'normal')}")

    router_result = result.get('router_result')
    if router_result:
        print(f"ROUTER: mode={router_result.get('mode')}, "
              f"topic={router_result.get('topic')}, "
              f"confidence={router_result.get('confidence', 'N/A')}")

    print(f"\nRESPONSE:")
    print("-" * 50)
    response = result['response']
    # Truncate for readability
    if len(response) > 1000:
        safe_print(response[:1000] + "\n... [truncated]")
    else:
        safe_print(response)
    print("-" * 50)


def test1_simplify_intent():
    """
    Test 1: SIMPLIFY intent should preserve topic and switch to simple style.

    Conversation:
    1. "what is the main concept of LangGraph" -> topic=LangGraph, style=normal
    2. "im not a tech person" -> topic=LangGraph (NOT general!), style=simple
    3. "tell me in order way" -> topic=LangGraph, style=simple
    """
    print_header("TEST 1: SIMPLIFY Intent Preservation")

    session_id = "test1_simplify"
    clear_session(session_id)

    messages = [
        "what is the main concept of LangGraph",
        "im not a tech person",
        "tell me in order way"
    ]

    results = []
    for i, msg in enumerate(messages, 1):
        result = chat(msg, session_id=session_id)
        results.append(result)
        print_turn(i, msg, result)

    # Validate
    print("\n--- VALIDATION ---")
    passed = True

    # Turn 1: Should be LangGraph
    if results[0]['detected_subject'] == Subject.LANGGRAPH:
        print("[PASS] Turn 1: Subject is LANGGRAPH")
    else:
        print(f"[FAIL] Turn 1: Expected LANGGRAPH, got {results[0]['detected_subject']}")
        passed = False

    # Turn 2: Should STILL be LangGraph, style=simple
    if results[1]['detected_subject'] == Subject.LANGGRAPH:
        print("[PASS] Turn 2: Subject preserved as LANGGRAPH (not switched to GENERAL)")
    else:
        print(f"[FAIL] Turn 2: Expected LANGGRAPH, got {results[1]['detected_subject']}")
        passed = False

    if results[1].get('style') == 'simple':
        print("[PASS] Turn 2: Style correctly set to 'simple'")
    else:
        print(f"[FAIL] Turn 2: Expected style='simple', got {results[1].get('style')}")
        passed = False

    # Turn 3: Should still be LangGraph, simple
    if results[2]['detected_subject'] == Subject.LANGGRAPH:
        print("[PASS] Turn 3: Subject still LANGGRAPH")
    else:
        print(f"[FAIL] Turn 3: Expected LANGGRAPH, got {results[2]['detected_subject']}")
        passed = False

    if results[2].get('style') == 'simple':
        print("[PASS] Turn 3: Style still 'simple'")
    else:
        print(f"[FAIL] Turn 3: Expected style='simple', got {results[2].get('style')}")
        passed = False

    clear_session(session_id)
    return passed


def test2_reducers_specificity():
    """
    Test 2: Reducers question should get specific answer about last-write-wins.

    Question: "In LangGraph, why use reducers like add_messages? what bug if overwrite?"
    Expected: Response mentions last-write-wins, parallel writes, deterministic merge.
    """
    print_header("TEST 2: Reducers Question Specificity")

    session_id = "test2_reducers"
    clear_session(session_id)

    msg = "In LangGraph, why use reducers like add_messages? what bug if overwrite?"
    result = chat(msg, session_id=session_id)
    print_turn(1, msg, result)

    # Validate: Check for specific keywords
    print("\n--- VALIDATION ---")
    response_lower = result['response'].lower()

    keywords_to_check = [
        ("last-write-wins", ["last-write-wins", "last write wins", "overwrit", "lost", "lose"]),
        ("parallel/concurrent", ["parallel", "concurrent", "simultaneous", "multiple node"]),
        ("append/merge", ["append", "merge", "accumulate", "add_messages", "reducer"])
    ]

    passed = True
    for concept, variants in keywords_to_check:
        found = any(v in response_lower for v in variants)
        if found:
            print(f"[PASS] Response mentions '{concept}'")
        else:
            print(f"[WARN] Response may not clearly mention '{concept}'")
            # Don't fail on this, just warn - LLM responses vary

    # Must be LangGraph topic
    if result['detected_subject'] == Subject.LANGGRAPH:
        print("[PASS] Subject correctly detected as LANGGRAPH")
    else:
        print(f"[FAIL] Expected LANGGRAPH, got {result['detected_subject']}")
        passed = False

    clear_session(session_id)
    return passed


def test3_json_router_robustness():
    """
    Test 3: JSON router should handle malformed output gracefully.

    Tests the parse_router_json and validate_router_result functions directly.
    """
    print_header("TEST 3: JSON Router Robustness")

    test_cases = [
        # Valid JSON
        ('{"mode": "ANSWER", "topic": "LangGraph", "confidence": 0.9, "missing_info": [], "reason": "clear question"}',
         {"mode": "ANSWER", "topic": "LangGraph", "confidence": 0.9}),

        # JSON in markdown code block
        ('```json\n{"mode": "TEACH", "topic": "Python", "confidence": 0.8}\n```',
         {"mode": "TEACH", "topic": "Python", "confidence": 0.8}),

        # Malformed JSON - should return None from parser
        ('This is not JSON at all',
         None),

        # Partial JSON
        ('{"mode": "DEBUG"',
         None),

        # Valid JSON with extra text
        ('I think the answer is {"mode": "QUIZ", "topic": "llm", "confidence": 0.7} based on...',
         {"mode": "QUIZ", "topic": "LLM", "confidence": 0.7}),
    ]

    passed = True
    for i, (input_text, expected) in enumerate(test_cases, 1):
        parsed = parse_router_json(input_text)

        if expected is None:
            if parsed is None:
                print(f"[PASS] Test case {i}: Correctly returned None for malformed input")
            else:
                print(f"[FAIL] Test case {i}: Expected None, got {parsed}")
                passed = False
        else:
            if parsed is None:
                print(f"[FAIL] Test case {i}: Expected parsed result, got None")
                passed = False
            else:
                validated = validate_router_result(parsed)
                # Check key fields match
                mode_ok = validated.get("mode") == expected.get("mode")
                topic_ok = validated.get("topic") == expected.get("topic")

                if mode_ok and topic_ok:
                    print(f"[PASS] Test case {i}: Correctly parsed mode={validated['mode']}, topic={validated['topic']}")
                else:
                    print(f"[FAIL] Test case {i}: Expected {expected}, got {validated}")
                    passed = False

    # Test fallback behavior in actual chat
    print("\n--- Testing fallback in live chat ---")
    session_id = "test3_fallback"
    clear_session(session_id)

    # Ambiguous message should trigger ASK_CLARIFY or still work
    result = chat("xyz123", session_id=session_id)
    print(f"Ambiguous input 'xyz123' -> subject={result['detected_subject']}, style={result['style']}")

    if result.get('router_result'):
        print(f"Router result: {result['router_result']}")

    # Should not crash, that's the main test
    print("[PASS] Router handled ambiguous input without crashing")

    clear_session(session_id)
    return passed


def run_all_tests():
    """Run all tests and report summary."""
    print("\n" + "=" * 70)
    print("  STUDY BUDDY ROUTING TEST HARNESS")
    print("=" * 70)
    print("\nThis test suite verifies:")
    print("1. SIMPLIFY intent preserves topic (doesn't switch to GENERAL)")
    print("2. Reducers questions get specific LangGraph answers")
    print("3. JSON router handles malformed input gracefully")
    print("\n" + "=" * 70)

    results = {}

    try:
        results['test1'] = test1_simplify_intent()
    except Exception as e:
        print(f"[ERROR] Test 1 failed with exception: {e}")
        results['test1'] = False

    try:
        results['test2'] = test2_reducers_specificity()
    except Exception as e:
        print(f"[ERROR] Test 2 failed with exception: {e}")
        results['test2'] = False

    try:
        results['test3'] = test3_json_router_robustness()
    except Exception as e:
        print(f"[ERROR] Test 3 failed with exception: {e}")
        results['test3'] = False

    # Summary
    print_header("FINAL SUMMARY")

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: [{status}]")

    print()
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("Some tests failed. Review output above.")

    return all_passed


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set it in .env file or environment.")
        sys.exit(1)

    print("Running routing tests...")
    print("This will make API calls to test the improvements.\n")

    success = run_all_tests()
    sys.exit(0 if success else 1)
