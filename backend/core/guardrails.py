"""Input and output guardrails for the LLM agent.

Implements OWASP LLM Top 10 defenses: prompt injection detection,
output filtering, PII scrubbing, and system prompt leakage prevention.
"""

import base64
import re
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""
    passed: bool
    reason: str = ""


# Regex patterns for known prompt injection attempts
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(in\s+)?developer\s+mode", re.IGNORECASE),
    re.compile(r"system\s+override", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+)?(system\s+)?prompt", re.IGNORECASE),
    re.compile(r"repeat\s+(the\s+)?(text|instructions?)\s+(above|before)", re.IGNORECASE),
    re.compile(r"what\s+(are|were)\s+your\s+(exact\s+)?instructions", re.IGNORECASE),
    re.compile(r"act\s+as\s+if\s+you.*(not|no)\s+bound", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"DAN\s+mode", re.IGNORECASE),
    re.compile(r"pretend\s+you\s+(are|have)\s+no\s+(restrictions?|rules?|limits?)", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(safety|previous|prior|your)", re.IGNORECASE),
    re.compile(r"bypass\s+(all\s+)?(safety|filters?|restrictions?|rules?)", re.IGNORECASE),
]

# Keywords for fuzzy (typoglycemia) matching
_FUZZY_KEYWORDS = [
    "ignore", "bypass", "override", "reveal", "delete",
    "system", "inject", "jailbreak", "prompt",
]

# Patterns that indicate system prompt leakage in output
_LEAKAGE_PATTERNS = [
    re.compile(r"you\s+are\s+icberg.*data\s+analysis", re.IGNORECASE),
    re.compile(r"##\s*(Rules|Dataset|Cache Context)", re.IGNORECASE),
    re.compile(r"ALWAYS\s+use\s+tools\s+to\s+compute", re.IGNORECASE),
    re.compile(r"SYSTEM_INSTRUCTIONS?:", re.IGNORECASE),
    re.compile(r"my\s+(system\s+)?instructions?\s+(are|say|tell)", re.IGNORECASE),
]

# PII / secret patterns to scrub from output
_PII_PATTERNS = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL_REDACTED]"),
    (re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"), "[PHONE_REDACTED]"),
    (re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"), "[CARD_REDACTED]"),
    (re.compile(r"(?:api[_-]?key|secret|token|password)\s*[:=]\s*\S+", re.IGNORECASE), "[SECRET_REDACTED]"),
    (re.compile(r"/(?:home|app|usr|etc|var|tmp)/[\w/.]+", re.IGNORECASE), "[PATH_REDACTED]"),
]

# Zero-width and invisible Unicode characters
_INVISIBLE_CHARS = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\u2060\u2061\u2062\u2063\u2064\ufeff]"
)

MAX_OUTPUT_LENGTH = 10_000


def _is_typo_variant(word: str, target: str) -> bool:
    """Check if a word is a typoglycemia variant of a target keyword."""
    if len(word) != len(target) or len(word) < 4:
        return False
    if word[0] != target[0] or word[-1] != target[-1]:
        return False
    return sorted(word[1:-1]) == sorted(target[1:-1]) and word != target


def _looks_like_base64_injection(text: str) -> bool:
    """Detect base64-encoded prompt injection payloads."""
    b64_pattern = re.compile(r"[A-Za-z0-9+/]{20,}={0,2}")
    for match in b64_pattern.finditer(text):
        try:
            decoded = base64.b64decode(match.group()).decode("utf-8", errors="ignore").lower()
            if any(kw in decoded for kw in ["ignore", "system", "prompt", "override", "bypass"]):
                return True
        except Exception:
            continue
    return False


class InputGuard:
    """Validates user input before it reaches the LLM."""

    def check(self, text: str) -> GuardrailResult:
        """Run all input checks. Returns GuardrailResult."""

        # Length check (defense-in-depth, Pydantic also enforces)
        if len(text) > 2000:
            return GuardrailResult(False, "Message too long")

        # Strip invisible unicode before pattern matching
        clean = _INVISIBLE_CHARS.sub("", text)

        # Regex injection patterns
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(clean):
                logger.warning("guardrail.input_blocked", reason="injection_pattern",
                               pattern=pattern.pattern[:40])
                return GuardrailResult(False, "prompt_injection_detected")

        # Typoglycemia (fuzzy) matching
        words = re.findall(r"\b\w+\b", clean.lower())
        for word in words:
            for keyword in _FUZZY_KEYWORDS:
                if _is_typo_variant(word, keyword):
                    logger.warning("guardrail.input_blocked", reason="typo_injection",
                                   word=word, target=keyword)
                    return GuardrailResult(False, "prompt_injection_detected")

        # Base64 encoded payloads
        if _looks_like_base64_injection(clean):
            logger.warning("guardrail.input_blocked", reason="base64_injection")
            return GuardrailResult(False, "encoded_injection_detected")

        return GuardrailResult(True)


class OutputGuard:
    """Validates and sanitizes LLM output before returning to the user."""

    def check(self, text: str) -> tuple[str, GuardrailResult]:
        """Validate and sanitize output. Returns (cleaned_text, result)."""
        if not text:
            return text, GuardrailResult(True)

        # System prompt leakage detection
        for pattern in _LEAKAGE_PATTERNS:
            if pattern.search(text):
                logger.warning("guardrail.output_blocked", reason="prompt_leakage",
                               pattern=pattern.pattern[:40])
                return (
                    "I'm unable to share that information. How can I help you with the Titanic dataset?",
                    GuardrailResult(False, "system_prompt_leakage"),
                )

        # PII / secret scrubbing
        cleaned = text
        scrubbed = False
        for pattern, replacement in _PII_PATTERNS:
            if pattern.search(cleaned):
                cleaned = pattern.sub(replacement, cleaned)
                scrubbed = True

        if scrubbed:
            logger.warning("guardrail.output_scrubbed", reason="pii_detected")

        # Length cap
        if len(cleaned) > MAX_OUTPUT_LENGTH:
            cleaned = cleaned[:MAX_OUTPUT_LENGTH] + "\n\n[Response truncated]"
            logger.warning("guardrail.output_truncated", original_len=len(text))

        return cleaned, GuardrailResult(True)
