"""Personal data redaction (GDPR, LLM02).

Replaces personal data with a typed placeholder: [EMAIL], [PHONE], [IBAN],
[CARD], [IP]. Pattern-based, so it finds identifiers, not names: "Inès Moreau"
stays. That limit is deliberate: catching names needs a NER model, and a
false positive in a Kestrel document ("Owner", "Business") would break answers.

Where it is applied:
- every question and answer stored in a trace (always);
- the evidence and the question sent to the LLM provider
  (REDACT_LLM_INPUT, on by default).

Role mailboxes (billing@, privacy@, support@...) are not personal data and are
kept, otherwise "where do I send a refund request?" could not be answered.
"""

from __future__ import annotations

import os
import re
from collections import Counter

ROLE_MAILBOXES = {
    "billing", "privacy", "support", "help", "info", "contact", "sales",
    "security", "noreply", "no-reply", "admin", "legal", "dpo", "hr", "jobs",
}

EMAIL = re.compile(r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,4})?\b")
CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
# International (+33 6 12 34 56 78) or national with separators (06 12 34 56 78,
# 555-123-4567). Bare digit runs are left alone: they are usually codes.
PHONE = re.compile(
    r"(?<![\w+])(?:\+\d{1,3}[ .-]?(?:\(0\)[ .-]?)?\d{1,4}(?:[ .-]?\d{2,4}){2,4}"
    r"|\(?0\d{1,3}\)?(?:[ .-]\d{2,4}){2,4})(?!\w)"
)


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


ALL_KINDS = ("email", "iban", "card", "ip", "phone")
# IP addresses in documents are usually servers (Kestrel's allowlist), not
# people: they are redacted in traces, not in what the LLM needs to answer.
LLM_INPUT_KINDS = ("email", "iban", "card", "phone")


def redact(text: str, kinds=ALL_KINDS) -> tuple[str, Counter]:
    """Return (redacted text, count per kind of personal data)."""
    if not text:
        return text, Counter()

    found: Counter = Counter()

    def email(match: re.Match) -> str:
        if match.group(1).lower() in ROLE_MAILBOXES:
            return match.group(0)
        found["email"] += 1
        return "[EMAIL]"

    def iban(match: re.Match) -> str:
        found["iban"] += 1
        return "[IBAN]"

    def card(match: re.Match) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            found["card"] += 1
            return "[CARD]"
        return match.group(0)

    def ip(match: re.Match) -> str:
        found["ip"] += 1
        return "[IP]"

    def phone(match: re.Match) -> str:
        if len(re.sub(r"\D", "", match.group(0))) < 9:
            return match.group(0)
        found["phone"] += 1
        return "[PHONE]"

    # Order matters: an IBAN or a card number must not be eaten as a phone number.
    for kind, pattern, replace in (
        ("email", EMAIL, email),
        ("iban", IBAN, iban),
        ("card", CARD, card),
        ("ip", IPV4, ip),
        ("phone", PHONE, phone),
    ):
        if kind in kinds:
            text = pattern.sub(replace, text)
    return text, found


def redact_text(text: str) -> str:
    return redact(text)[0]


def redact_for_llm(text: str) -> tuple[str, Counter]:
    if not redact_llm_input():
        return text, Counter()
    return redact(text, LLM_INPUT_KINDS)


def redact_llm_input() -> bool:
    return os.getenv("REDACT_LLM_INPUT", "true").strip().lower() not in {"0", "false", "no", "off"}
