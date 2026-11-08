"""Concept graph schema: which entity and relation types exist.

The schema is fixed (not open to whatever the LLM finds): a fixed list is easy to
validate and to query, and stops the LLM inventing 40 spellings of "owned by".
It was written for the Kestrel knowledge base: error codes, services, teams.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, Field

EntityType = Literal[
    "ErrorCode",  # KST-5031
    "Service",  # postroom, ledger, auth
    "Team",  # Messaging, Identity, Payments Core
    "Person",  # Inès Moreau
    "Channel",  # #oncall-messaging
    "Plan",  # Business, Enterprise
    "Feature",  # single sign-on, audit log, scheduled reports
    "Role",  # Owner, Admin, Editor, Viewer
]

ENTITY_TYPES: tuple[str, ...] = EntityType.__args__

# relation -> (allowed source types, allowed target types, meaning)
RELATION_SIGNATURES: dict[str, tuple[set[str], set[str], str]] = {
    "raised_by": ({"ErrorCode"}, {"Service"}, "the error is raised by the service"),
    "escalates_to": ({"ErrorCode"}, {"Team", "Service"}, "support escalates the error to this team (or to the team owning this service)"),
    "caused_by": ({"ErrorCode"}, {"ErrorCode"}, "the error is caused by another error"),
    "owned_by": ({"Service"}, {"Team"}, "the team owns the service"),
    "replaces": ({"Service"}, {"Service"}, "the first service replaced the second one"),
    "depends_on": ({"Service"}, {"Service"}, "the first service uses the second one"),
    "led_by": ({"Team"}, {"Person"}, "the person is the team's engineering lead"),
    "has_oncall_channel": ({"Team"}, {"Channel"}, "the team's on-call channel"),
    "covered_by": ({"Team"}, {"Team"}, "outside its hours, the first team is covered by the second"),
    "includes_feature": ({"Plan"}, {"Feature"}, "the plan includes the feature"),
    "provided_by": ({"Feature"}, {"Service"}, "the service provides the feature"),
    "requires_role": ({"Feature"}, {"Role"}, "using or managing the feature requires this role"),
}

RelationType = Literal[
    "raised_by",
    "escalates_to",
    "caused_by",
    "owned_by",
    "replaces",
    "depends_on",
    "led_by",
    "has_oncall_channel",
    "covered_by",
    "includes_feature",
    "provided_by",
    "requires_role",
]


class ExtractedEntity(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    type: EntityType
    aliases: list[str] = Field(default_factory=list, max_length=5)


class ExtractedRelation(BaseModel):
    source: str = Field(min_length=1, max_length=120)
    relation: RelationType
    target: str = Field(min_length=1, max_length=120)
    quote: str = Field(min_length=3, max_length=400)


def normalize_name(name: str, entity_type: str | None = None) -> str:
    """Canonical form of a name, used for entity resolution and matching.

    "The Messaging team" and "messaging" both become "messaging".
    """
    value = unicodedata.normalize("NFKC", name).casefold()
    value = value.strip().strip("`*\"'.:;,() ")
    value = re.sub(r"\s+", " ", value)

    if value.startswith("the "):
        value = value[4:]

    if entity_type == "Team" and value.endswith(" team"):
        value = value[: -len(" team")]
    if entity_type == "Service" and value.endswith(" service"):
        value = value[: -len(" service")]
    if entity_type == "Channel" and value and not value.startswith("#"):
        value = "#" + value

    return value.strip()


def normalize_text(text: str) -> str:
    """Lowercase, single spaces, no markdown emphasis: used to check that a quote
    is in a chunk. Models often drop the ** of "**Escalate:**" when copying."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("*", "").replace("`", "")
    return " ".join(text.split())
