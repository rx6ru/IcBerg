"""Contract tests for the LLM adapter (mocked, no real API calls)."""

import os
import pytest
from langchain_core.messages import HumanMessage, AIMessage

from backend.core.llm_adapter import LLMAdapter, LLMError, LLMUnavailableError


@pytest.fixture
def adapter():
    os.environ["CEREBRAS_API_KEY"] = "test-cerebras-key"
    os.environ["GROQ_API_KEY"] = "test-groq-key"
    os.environ["CEREBRAS_MODEL"] = "test-cerebras-model"
    os.environ["GROQ_MODEL"] = "test-groq-model"
    os.environ["LLM_TIMEOUT"] = "1"
    return LLMAdapter()


class TestLLMAdapterInit:

    def test_loads_keys(self, adapter):
        assert adapter.cerebras_key == "test-cerebras-key"
        assert adapter.groq_key == "test-groq-key"

    def test_primary_is_cerebras(self, adapter):
        from langchain_cerebras import ChatCerebras
        from langchain_core.runnables import RunnableWithFallbacks
        model = adapter.get_chat_model()
        assert isinstance(model, RunnableWithFallbacks)
        assert isinstance(model.runnable, ChatCerebras)
        assert model.runnable.model == "test-cerebras-model"
        assert model.runnable.temperature == 0.0
        assert model.runnable.max_tokens == 2048


class TestLLMFailover:

    def test_cerebras_success(self, adapter, mocker):
        mock_invoke = mocker.patch("langchain_cerebras.ChatCerebras.invoke")
        mock_invoke.return_value = AIMessage(content="Cerebras response")

        model = adapter.get_chat_model()
        response = model.invoke([HumanMessage(content="Hello")])

        assert response.content == "Cerebras response"
        mock_invoke.assert_called_once()

    def test_timeout_falls_back_to_groq(self, adapter, mocker):
        from httpx import ReadTimeout

        mocker.patch("langchain_cerebras.ChatCerebras.invoke", side_effect=ReadTimeout("Timeout"))
        mock_groq = mocker.patch("langchain_groq.ChatGroq.invoke")
        mock_groq.return_value = AIMessage(content="Groq response")

        response = adapter.invoke_with_failover([HumanMessage(content="Hello")])

        assert response.content == "Groq response"
        mock_groq.assert_called_once()

    def test_4xx_does_not_fallback(self, adapter, mocker):
        from httpx import HTTPStatusError, Request, Response

        resp = Response(status_code=400, request=Request("POST", "http://test"))
        err = HTTPStatusError("Bad Request", request=resp.request, response=resp)

        mocker.patch("langchain_cerebras.ChatCerebras.invoke", side_effect=err)
        mock_groq = mocker.patch("langchain_groq.ChatGroq.invoke")

        with pytest.raises(LLMError) as exc:
            adapter.invoke_with_failover([HumanMessage(content="Hello")])

        assert "400" in str(exc.value)
        mock_groq.assert_not_called()

    def test_both_fail_raises_unavailable(self, adapter, mocker):
        from httpx import ReadTimeout

        mocker.patch("langchain_cerebras.ChatCerebras.invoke", side_effect=ReadTimeout("T1"))
        mocker.patch("langchain_groq.ChatGroq.invoke", side_effect=ReadTimeout("T2"))

        with pytest.raises(LLMUnavailableError):
            adapter.invoke_with_failover([HumanMessage(content="Hello")])
