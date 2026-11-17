"""Offline fixture catalog. Topics are generic categories, never a commercial brand."""

from __future__ import annotations

from typing import Any

# Offering names are role labels in a category, not fake product brands.
INCUMBENT = "Incumbent suite"
SPECIALIST = "Lightweight specialist"
OPEN_TOOL = "Open collaboration tool"


def _offering(
    name: str,
    positioning: str,
    strengths: list[str],
    gaps: list[str],
) -> dict[str, Any]:
    return {
        "name": name,
        "positioning": positioning,
        "strengths": strengths,
        "gaps": gaps,
    }


CATALOG: dict[str, dict[str, Any]] = {
    "research notebooks for distributed teams": {
        "keywords": ("research notebook", "interview notes", "distributed team"),
        "category": "research-notebooks",
        "problem_context": (
            "Distributed product teams lose interview and discovery context between "
            "tickets, design notes, and handoffs."
        ),
        "user_need_findings": [
            "Capture interview notes with shared tags",
            "Search findings across sessions without a single-vendor lock-in",
            "Hand a structured brief to engineering without rebuilding slides",
        ],
        "offerings": [
            _offering(
                INCUMBENT,
                "All-in-one workspace with documents and tasks",
                ["Familiar to large teams", "Broad sharing"],
                ["Weak research tagging", "Hard to compare alternatives"],
            ),
            _offering(
                SPECIALIST,
                "Notebook focused on qualitative research",
                ["Fast capture", "Tagging"],
                ["Limited competitive-analysis tables"],
            ),
            _offering(
                OPEN_TOOL,
                "Shared docs with comments",
                ["Low friction", "Exportable"],
                ["No structured competitor comparison"],
            ),
        ],
        "recommendations": [
            "Treat research notes as a structured artifact, not a slide deck",
            "Keep competitor rows as role labels (incumbent / specialist / open) until a real shortlist exists",
            "Require problem/context, user needs, comparison, and implications before a go-to-market discussion",
        ],
        "sources": ["fixture:research-notebooks"],
    },
    "team knowledge bases": {
        "keywords": ("knowledge base", "wiki", "internal docs"),
        "category": "team-knowledge-bases",
        "problem_context": (
            "Product teams need a durable place for decisions and research so new "
            "members do not re-ask the same questions."
        ),
        "user_need_findings": [
            "Find the latest decision without pinging the author",
            "Link a finding to the interview evidence that produced it",
            "Keep access scoped to the team that owns the notes",
        ],
        "offerings": [
            _offering(
                INCUMBENT,
                "Enterprise wiki bundled with the company suite",
                ["SSO and admin controls", "Existing adoption"],
                ["Search quality varies", "Research structure is ad hoc"],
            ),
            _offering(
                SPECIALIST,
                "Team wiki with trees and backlinks",
                ["Fast linking", "Local ownership"],
                ["Weak competitive-analysis templates"],
            ),
            _offering(
                OPEN_TOOL,
                "Plain markdown in a shared repo",
                ["Reviewable in git", "No extra vendor"],
                ["Discovery depends on folder discipline"],
            ),
        ],
        "recommendations": [
            "Store research briefs next to the decision they informed",
            "Tag pages by scene and risk so evaluation datasets can reuse them later",
            "Do not copy another tenant's briefs into this knowledge base",
        ],
        "sources": ["fixture:team-knowledge-bases"],
    },
    "calendar coordination for product teams": {
        "keywords": ("calendar", "scheduling", "coordination"),
        "category": "calendar-coordination",
        "problem_context": (
            "Research interviews and competitor reviews slip when scheduling is ad hoc "
            "across time zones."
        ),
        "user_need_findings": [
            "Book interviews without leaking private calendar details to a model",
            "See which competitive-review slots are already claimed",
            "Keep a record of who attended a discovery session",
        ],
        "offerings": [
            _offering(
                INCUMBENT,
                "Company calendar with meeting links",
                ["Already deployed", "Busy/free sharing"],
                ["No research-brief template on the invite"],
            ),
            _offering(
                SPECIALIST,
                "Scheduling pages for external interviews",
                ["Fewer email round-trips"],
                ["Weak link back to the research artifact"],
            ),
            _offering(
                OPEN_TOOL,
                "Shared availability spreadsheet plus a video link",
                ["Works immediately"],
                ["No audit trail", "Easy to double-book"],
            ),
        ],
        "recommendations": [
            "Attach the research brief id to every interview event",
            "Keep calendar credentials in the vault; never paste them into prompts",
            "Prefer a canary rollout of any scheduling helper before full traffic",
        ],
        "sources": ["fixture:calendar-coordination"],
    },
}


def generic_entry(topic: str) -> dict[str, Any]:
    cleaned = topic.strip() or "unspecified topic"
    return {
        "keywords": (),
        "category": "generic",
        "problem_context": (
            f"Teams exploring {cleaned} need a shared view of the problem, who it is for, "
            "and how current options compare."
        ),
        "user_need_findings": [
            f"State the job to be done for {cleaned} in one paragraph",
            "List the users who feel the pain most often",
            "Separate must-have needs from nice-to-have comparisons",
        ],
        "offerings": [
            _offering(
                INCUMBENT,
                f"Default suite already used around {cleaned}",
                ["Familiar", "Broad coverage"],
                ["Shallow fit for this job"],
            ),
            _offering(
                SPECIALIST,
                f"Focused tool for {cleaned}",
                ["Deeper fit"],
                ["Narrower ecosystem"],
            ),
            _offering(
                OPEN_TOOL,
                "Composable open workflow",
                ["Portable artifacts"],
                ["More assembly required"],
            ),
        ],
        "recommendations": [
            f"Write a structured brief for {cleaned} before choosing a path",
            "Compare at least two role-level alternatives, not a single anecdote",
            "Record implications that a later evaluation gate can assert on",
        ],
        "sources": ["fixture:generic"],
    }


def resolve_entry(topic: str) -> dict[str, Any]:
    key = (topic or "").strip().lower()
    if key in CATALOG:
        return dict(CATALOG[key])
    for name, entry in CATALOG.items():
        if name in key or key in name:
            return dict(entry)
        if any(token in key for token in entry.get("keywords") or ()):
            return dict(entry)
    return generic_entry(topic)
