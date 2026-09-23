"""Deterministic guardrail stages for LLM applications.

Every stage is a small, composable class behind one interface:

    result = stage(text, context)   # -> StageResult(verdict, reason, ...)

Verdicts are "pass", "block", or "redact". All checks are regex/keyword
based, so the whole pipeline runs offline with zero model calls and fully
deterministic results.

Production swap: replace any stage with an LLM-as-judge implementation that
returns the same StageResult shape. The Pipeline and the eval harness do
not care how a stage decides.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

PASS, BLOCK, REDACT = "pass", "block", "redact"


@dataclass
class StageResult:
    """Outcome of one guardrail stage."""

    stage: str
    verdict: str  # PASS | BLOCK | REDACT
    reason: str
    latency_ms: float = 0.0
    details: Dict = field(default_factory=dict)


class GuardrailStage:
    """Base class: measure latency, delegate the decision to subclasses."""

    name = "base"

    def __call__(self, text: str, context: Optional[Dict] = None) -> StageResult:
        start = time.perf_counter()
        try:
            result = self.check(text or "", context or {})
        except Exception as exc:  # a crashing stage must not silently pass
            result = StageResult(self.name, BLOCK, f"stage error: {exc}")
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    def check(self, text: str, context: Dict) -> StageResult:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Input stages
# ---------------------------------------------------------------------------

PII_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "api_key": re.compile(
        r"\b(?:sk-[A-Za-z0-9_-]{8,}|xox[baprs]-[A-Za-z0-9-]+"
        r"|ghp_[A-Za-z0-9]{8,}|AKIA[0-9A-Z]{16})\b"
    ),
}


def _luhn_ok(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if len(digits) < 13:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def find_pii(text: str) -> List[Dict[str, str]]:
    """Return [{type, value}] for every PII match (credit cards Luhn-checked)."""
    hits = []
    for pii_type, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            value = match.group(0)
            if pii_type == "credit_card" and not _luhn_ok(value):
                continue
            hits.append({"type": pii_type, "value": value})
    return hits


def redact_pii(text: str) -> str:
    """Replace PII with typed placeholders, e.g. [REDACTED:EMAIL]."""

    def _sub(pii_type: str, pattern: re.Pattern) -> None:
        nonlocal text
        def repl(m: re.Match) -> str:
            if pii_type == "credit_card" and not _luhn_ok(m.group(0)):
                return m.group(0)
            return f"[REDACTED:{pii_type.upper()}]"
        text = pattern.sub(repl, text)

    for pii_type, pattern in PII_PATTERNS.items():
        _sub(pii_type, pattern)
    return text


class PIIDetectionStage(GuardrailStage):
    """Block (input) or redact (output) text containing emails, phones,
    SSNs, credit cards, or API keys."""

    name = "pii"

    def __init__(self, mode: str = "block"):
        if mode not in (BLOCK, REDACT):
            raise ValueError("mode must be 'block' or 'redact'")
        self.mode = mode

    def check(self, text: str, context: Dict) -> StageResult:
        hits = find_pii(text)
        if not hits:
            return StageResult(self.name, PASS, "no PII detected")
        types = sorted({h["type"] for h in hits})
        details = {"types": types, "count": len(hits)}
        if self.mode == BLOCK:
            return StageResult(
                self.name, BLOCK, f"PII detected: {', '.join(types)}", details=details
            )
        redacted = redact_pii(text)
        return StageResult(
            self.name, REDACT, f"PII redacted: {', '.join(types)}",
            details={**details, "redacted_text": redacted},
        )


INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|your)\s+(instructions|safety|guidelines|rules)", re.I),
    re.compile(r"\byou\s+are\s+now\b", re.I),
    re.compile(r"\bdo\s+anything\s+now\b|\bDAN\s+mode\b", re.I),
    re.compile(r"\bpretend\s+(you\s+are|to\s+be)\b", re.I),
    re.compile(r"\bjailbreak\b", re.I),
    re.compile(r"(^|\n)\s*(system|developer)\s*:", re.I),
    re.compile(r"\bdeveloper\s+mode\b", re.I),
    re.compile(r"\breveal\s+(your|the)\s+(system\s+prompt|instructions)\b", re.I),
    re.compile(r"\bbypass\s+(your|the)\s+(safety|filters|restrictions)\b", re.I),
    re.compile(r"\bunrestricted\s+(ai|assistant|model)\b", re.I),
]


class PromptInjectionStage(GuardrailStage):
    """Detect instruction-override phrases, role-play jailbreak markers,
    and fake system/developer role injections."""

    name = "prompt_injection"

    def check(self, text: str, context: Dict) -> StageResult:
        matched = [p.pattern for p in INJECTION_PATTERNS if p.search(text)]
        if not matched:
            return StageResult(self.name, PASS, "no injection patterns")
        return StageResult(
            self.name, BLOCK,
            f"prompt-injection pattern: {matched[0]}",
            details={"patterns": matched},
        )


TOXIC_KEYWORDS = [
    r"\bidiot\b", r"\bstupid\b", r"\bdumb\b", r"\bmoron\b", r"\bworthless\b",
    r"\bi\s+hate\s+you\b", r"\bkill\s+yourself\b", r"\bshut\s+up\b",
    r"\bfuck\b", r"\bshit\b", r"\bbitch\b", r"\basshole\b", r"\bdamn\s+you\b",
]
TOXIC_PATTERN = re.compile("|".join(TOXIC_KEYWORDS), re.I)


class ToxicityStage(GuardrailStage):
    """Keyword-based toxicity filter (offline stand-in for a moderation
    classifier; swap for the real classifier API in production)."""

    name = "toxicity"

    def check(self, text: str, context: Dict) -> StageResult:
        hits = sorted(set(TOXIC_PATTERN.findall(text)))
        if not hits:
            return StageResult(self.name, PASS, "no toxic keywords")
        return StageResult(
            self.name, BLOCK, f"toxic language: {', '.join(hits)}",
            details={"keywords": hits},
        )


class LengthPolicyStage(GuardrailStage):
    """Enforce max input length / line-count policy (DoS + cost guard)."""

    name = "length_policy"

    def __init__(self, max_chars: int = 4000, max_lines: int = 200):
        self.max_chars = max_chars
        self.max_lines = max_lines

    def check(self, text: str, context: Dict) -> StageResult:
        if len(text) > self.max_chars:
            return StageResult(
                self.name, BLOCK,
                f"input too long: {len(text)} chars > {self.max_chars}",
                details={"chars": len(text)},
            )
        lines = text.count("\n") + 1
        if lines > self.max_lines:
            return StageResult(
                self.name, BLOCK,
                f"too many lines: {lines} > {self.max_lines}",
                details={"lines": lines},
            )
        return StageResult(self.name, PASS, "within length policy")


# ---------------------------------------------------------------------------
# Output stages
# ---------------------------------------------------------------------------

REFUSAL_PATTERN = re.compile(
    r"\bi\s+(can't|cannot|am\s+unable\s+to|don't\s+have\s+the\s+ability\s+to)\s+"
    r"(help|assist|comply|do\s+that)|i['’]m\s+unable\s+to|as\s+an\s+ai\b"
    r"|i\s+must\s+decline|not\s+able\s+to\s+assist",
    re.I,
)


def is_refusal(text: str) -> bool:
    return bool(REFUSAL_PATTERN.search(text or ""))


class RefusalConsistencyStage(GuardrailStage):
    """Flag when a response refuses when it should comply, or complies
    when the (blocked) prompt required a refusal. The expectation comes
    from context["should_refuse"]; without it the stage passes."""

    name = "refusal_consistency"

    def check(self, text: str, context: Dict) -> StageResult:
        should_refuse = context.get("should_refuse")
        if should_refuse is None:
            return StageResult(self.name, PASS, "no refusal expectation set")
        refused = is_refusal(text)
        if should_refuse and not refused:
            return StageResult(
                self.name, BLOCK,
                "response should refuse (input was blocked) but complies",
            )
        if not should_refuse and refused:
            return StageResult(
                self.name, BLOCK, "over-refusal: benign prompt got a refusal"
            )
        return StageResult(self.name, PASS, "refusal behavior consistent")


URL_PATTERN = re.compile(r"https?://[^\s)>\]]+")
CITATION_PATTERN = re.compile(
    r"\bet\s+al\.?|\([A-Z][A-Za-z-]+,?\s*20\d{2}\)|according\s+to\b", re.I
)


class HallucinationHeuristicStage(GuardrailStage):
    """Heuristic flags for fabricated authority: URLs and academic-style
    citations in a response cannot be verified offline, so flag them for
    review. In production, pair with a retrieval/verification step."""

    name = "hallucination_heuristic"

    def check(self, text: str, context: Dict) -> StageResult:
        urls = URL_PATTERN.findall(text or "")
        citations = CITATION_PATTERN.findall(text or "")
        flags = []
        if urls:
            flags.append(f"unverifiable URLs: {urls[:3]}")
        if citations:
            flags.append("unverifiable citation phrasing")
        if not flags:
            return StageResult(self.name, PASS, "no fabricated-authority signals")
        return StageResult(
            self.name, BLOCK, "; ".join(flags),
            details={"urls": urls[:5], "citation_match": bool(citations)},
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineReport:
    """Full result of running a prompt (+ optional response) through the pipeline."""

    verdict: str  # PASS | BLOCK | REDACT
    stage_results: List[StageResult]
    total_ms: float = 0.0
    redacted_response: Optional[str] = None

    @property
    def latencies_ms(self) -> Dict[str, float]:
        return {r.stage: r.latency_ms for r in self.stage_results}

    @property
    def blocking_stage(self) -> Optional[str]:
        for r in self.stage_results:
            if r.verdict == BLOCK:
                return r.stage
        return None


class GuardrailPipeline:
    """Chain input stages over the prompt, then output stages over the
    response. Stops at the first BLOCK (short-circuit, fail-fast)."""

    def __init__(
        self,
        input_stages: Optional[List[GuardrailStage]] = None,
        output_stages: Optional[List[GuardrailStage]] = None,
    ):
        self.input_stages = input_stages or []
        self.output_stages = output_stages or []

    def run(
        self,
        prompt: str,
        response: Optional[str] = None,
        context: Optional[Dict] = None,
    ) -> PipelineReport:
        context = dict(context or {})
        results: List[StageResult] = []
        start = time.perf_counter()

        for stage in self.input_stages:
            result = stage(prompt, context)
            results.append(result)
            if result.verdict == BLOCK:
                return PipelineReport(BLOCK, results, _ms(start))

        redacted_response = response
        output_text = response
        for stage in self.output_stages:
            result = stage(output_text or "", context)
            results.append(result)
            if result.verdict == REDACT:
                output_text = result.details.get("redacted_text", output_text)
                redacted_response = output_text
            elif result.verdict == BLOCK:
                return PipelineReport(
                    BLOCK, results, _ms(start), redacted_response=redacted_response
                )

        verdict = REDACT if any(r.verdict == REDACT for r in results) else PASS
        return PipelineReport(verdict, results, _ms(start),
                              redacted_response=redacted_response)


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def build_default_pipeline() -> GuardrailPipeline:
    """Standard layout: strict input checks, scrub-then-verify on output."""
    return GuardrailPipeline(
        input_stages=[
            PIIDetectionStage(mode="block"),
            PromptInjectionStage(),
            ToxicityStage(),
            LengthPolicyStage(),
        ],
        output_stages=[
            PIIDetectionStage(mode="redact"),  # PII scrubbing on the way out
            RefusalConsistencyStage(),
            HallucinationHeuristicStage(),
        ],
    )
