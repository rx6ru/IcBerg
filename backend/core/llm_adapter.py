"""LLM adapter with Cerebras → Groq failover.

Cerebras is the primary (fast inference). On timeout or 5xx, falls back to Groq.
4xx errors fail immediately — no point retrying a bad request on a different provider.
"""

import os

import structlog
from httpx import HTTPStatusError, ReadTimeout
from langchain_cerebras import ChatCerebras
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq

logger = structlog.get_logger(__name__)


class LLMError(Exception):
    """Non-retryable LLM error (e.g. 4xx bad request)."""
    pass


class LLMUnavailableError(Exception):
    """Both providers are down or timing out."""
    pass


class LLMAdapter:
    """Wraps Cerebras + Groq with automatic failover."""

    def __init__(self):
        self.cerebras_key = os.environ.get("CEREBRAS_API_KEY", "")
        self.groq_key = os.environ.get("GROQ_API_KEY", "")

        self.cerebras_model_name = os.environ.get("CEREBRAS_MODEL", "gpt-oss-120b")
        self.groq_model_name = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

        self.temperature = float(os.environ.get("LLM_TEMPERATURE", "0"))
        self.max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "2048"))
        self.timeout = int(os.environ.get("LLM_TIMEOUT", "30"))

        self.primary_llm = ChatCerebras(
            api_key=self.cerebras_key,
            model=self.cerebras_model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )

        self.fallback_llm = ChatGroq(
            api_key=self.groq_key,
            model=self.groq_model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )

    def get_chat_model(self) -> BaseChatModel:
        """Return the primary model instance.

        Returns:
            ChatCerebras instance. For failover, use invoke_with_failover() instead.
        """
        return self.primary_llm

    def is_healthy(self) -> bool:
        """Check if at least one provider has a key configured.

        Returns:
            True if either Cerebras or Groq API key is set.
        """
        return bool(self.cerebras_key) or bool(self.groq_key)

    def invoke_with_failover(self, messages: list[BaseMessage]) -> BaseMessage:
        """Try Cerebras first, fall back to Groq on timeout/5xx.

        Args:
            messages: List of LangChain message objects to send.

        Returns:
            AI response message from whichever provider succeeds.

        Raises:
            LLMError: If Cerebras returns a 4xx (no fallback attempted).
            LLMUnavailableError: If both providers fail.
        """
        logger.debug("llm.invoke", provider="cerebras", model=self.cerebras_model_name)

        try:
            return self.primary_llm.invoke(messages)

        except HTTPStatusError as e:
            if 400 <= e.response.status_code < 500:
                logger.error("llm.4xx", status=e.response.status_code)
                raise LLMError(f"Cerebras API rejected request ({e.response.status_code}): {e}")

            logger.warning("llm.5xx_fallback", status=e.response.status_code)

        except ReadTimeout:
            logger.warning("llm.timeout_fallback", threshold=self.timeout)

        except Exception as e:
            logger.warning("llm.unknown_fallback", error=str(e))

        # Fallback to Groq
        logger.info("llm.groq_fallback", model=self.groq_model_name)
        try:
            response = self.fallback_llm.invoke(messages)
            logger.info("llm.groq_ok")
            return response

        except Exception as e:
            logger.error("llm.both_failed", error=str(e))
            raise LLMUnavailableError(f"Both primary and fallback LLMs failed: {e}")
