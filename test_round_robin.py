import os
from unittest.mock import MagicMock

from backend.core.llm_adapter import LLMAdapter


def test_round_robin():
    # Force both keys to exist
    os.environ["CEREBRAS_API_KEY"] = "fake"
    os.environ["GROQ_API_KEY"] = "fake"

    adapter = LLMAdapter()

    # Mock the actual models internally
    adapter.primary_llm = MagicMock()
    adapter.primary_llm.with_fallbacks.return_value = "cerebras_with_fallback"

    adapter.fallback_llm = MagicMock()
    adapter.fallback_llm.with_fallbacks.return_value = "groq_with_fallback"

    # Call 1
    m1 = adapter.get_chat_model()
    print("Call 1:", m1)

    # Call 2
    m2 = adapter.get_chat_model()
    print("Call 2:", m2)

    # Call 3
    m3 = adapter.get_chat_model()
    print("Call 3:", m3)

    assert m1 != m2, "Should alternate!"
    assert m1 == m3, "Should repeat after 2!"

if __name__ == "__main__":
    test_round_robin()
    print("Round-Robin test passed!")
