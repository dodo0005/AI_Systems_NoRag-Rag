"""Prompt injection scanner (LLM01 + LLM05).

Rules, not a model: each rule is a regular expression with a name and a
severity, so a finding can be explained ("matched 'ignore previous
instructions' at line 12") and a false positive can be fixed by editing a rule.
No filter catches every injection; the scanner is one layer, next to the
prompt that treats evidence as data and the trace of every answer's chunks.

Three places use it:
- upload: a document with a high-severity finding is not indexed, it is moved
  to quarantine/<corpus>/ with a JSON report next to it;
- question time: retrieved chunks with a high-severity finding are withheld
  from the prompt (and listed in the trace);
- the question itself: a high-severity question is refused (outcome "blocked").
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

HIGH, MEDIUM = "high", "medium"

# (name, severity, pattern). Patterns run on a normalised copy of the text:
# NFKC, casefolded, zero-width characters removed, whitespace collapsed.
RULES: list[tuple[str, str, re.Pattern]] = [
    ("override_instructions", HIGH, re.compile(
        r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|your|system|these)\b[^.\n]{0,30}\b(instructions?|prompts?|rules?|directions?|guidelines?)\b")),
    ("addressed_to_the_model", HIGH, re.compile(
        r"\bif you are an? (ai|llm|language model|assistant|chatbot|coding agent|agent)\b")),
    ("new_role", HIGH, re.compile(
        r"\byou are now (an? |the |in )?(\w+ )?(mode|assistant|ai|model|agent|bot|unrestricted|jailbroken|dan)\b"
        r"|\bfrom now on,? you (will|must|should) (ignore|only|always|act|answer|respond|reply)\b"
        r"|\bact as (an?|the) (unfiltered|unrestricted|jailbroken|different)\b")),
    ("reveal_prompt", HIGH, re.compile(
        r"\b(reveal|print|show|repeat|output|leak)\b[^.\n]{0,30}\b(system prompt|your instructions|hidden (prompt|instructions)|initial prompt)\b")),
    ("fake_system_message", HIGH, re.compile(
        r"(^|\n)\s*(system|assistant)\s*:|<\s*/?\s*(system|im_start|im_end)\s*>|\[\s*(system|inst)\s*\]")),
    ("markdown_image_exfiltration", HIGH, re.compile(
        r"!\[[^\]]*\]\(\s*https?://[^)\s]*[?&][^)\s]*=")),
    ("tool_or_action_request", MEDIUM, re.compile(
        r"\b(send|email|post|upload|forward)\b[^.\n]{0,40}\b(to|at)\b[^.\n]{0,40}(https?://|@)")),
    ("answer_override", MEDIUM, re.compile(
        r"\b(always|only) (answer|reply|respond|say)\b[^.\n]{0,40}\b(with|that)\b")),
]

ZERO_WIDTH = re.compile(r"[​-‏⁠-⁤﻿]")
HTML_COMMENT = re.compile(r"<!--(.*?)-->", re.S)


@dataclass
class Finding:
    rule: str
    severity: str
    line: int
    excerpt: str


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = ZERO_WIDTH.sub("", text)
    return text.casefold()


def scan(text: str) -> list[Finding]:
    """Every rule match in the text, with its line number."""
    findings: list[Finding] = []
    if not text:
        return findings

    raw = unicodedata.normalize("NFKC", text)
    norm = normalise(text)

    for name, severity, pattern in RULES:
        for match in pattern.finditer(norm):
            line = norm.count("\n", 0, match.start()) + 1
            excerpt = " ".join(norm[match.start(): match.end()].split())[:120]
            findings.append(Finding(name, severity, line, excerpt))

    # Hidden text is a signal on its own: zero-width characters, and HTML
    # comments that contain words (they are not displayed, but the LLM reads them).
    hidden = len(ZERO_WIDTH.findall(raw))
    if hidden >= 3:
        findings.append(Finding("zero_width_characters", MEDIUM, 1, f"{hidden} zero-width characters"))
    for match in HTML_COMMENT.finditer(raw):
        body = " ".join(match.group(1).split())
        if len(body.split()) >= 4:
            line = raw.count("\n", 0, match.start()) + 1
            findings.append(Finding("hidden_html_comment", MEDIUM, line, body[:120]))

    return findings


def is_high_risk(findings: list[Finding]) -> bool:
    return any(f.severity == HIGH for f in findings)


def as_dicts(findings: list[Finding]) -> list[dict]:
    return [asdict(f) for f in findings]
