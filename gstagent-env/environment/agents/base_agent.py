"""
Base Agent — abstract interface for all specialist agents.

Each agent has a role, a system prompt, and can process observations
to produce actions. Agents communicate via the orchestrator's message bus.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Shared grounding clause — injected into every agent's system prompt
# This is the final 10% gap between "rarely hallucinates" and "provably cannot"
GROUNDING_CLAUSE = """
CRITICAL GROUNDING RULE:
If the retrieved GST knowledge does not contain the specific rule, circular,
percentage, amount, or date needed to answer the question, respond with:
"I don't have a verified source for this in the current knowledge base.
Please consult CBIC directly or refer to the GST portal at cbic-gst.gov.in."
Do NOT infer, extrapolate, or recall training knowledge for GST compliance answers.
Only cite rules, sections, circulars, and numbers that appear in the provided context.
"""


@dataclass
class AgentMessage:
    """Message passed between agents via the orchestrator."""

    sender: str
    content: str
    action: dict | None = None
    metadata: dict = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base for all specialist agents."""

    def __init__(self, name: str, role: str, model: str | None = None):
        from environment.config import MODEL_NAME
        self.name = name
        self.role = role
        self.model = model or MODEL_NAME
        self.message_history: list[dict] = []
        self.actions_taken: int = 0

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the agent's system prompt."""

    @abstractmethod
    def process(
        self, observation: dict, context: list[AgentMessage]
    ) -> list[AgentMessage]:
        """Process an observation and return messages/actions."""

    def build_messages(self, observation: dict, context: list[AgentMessage]) -> list[dict]:
        """Build LLM message list from observation and context."""
        messages = [{"role": "system", "content": self.get_system_prompt()}]

        # Add context from other agents
        for msg in context[-10:]:  # Last 10 messages to avoid overflow
            messages.append({
                "role": "user",
                "content": f"[{msg.sender}]: {msg.content}",
            })

        # Add current observation
        messages.append({
            "role": "user",
            "content": f"Current observation:\n{json.dumps(observation, indent=2, default=str)[:3000]}",
        })

        return messages

    def create_action_message(
        self, content: str, action: dict | None = None, **metadata
    ) -> AgentMessage:
        """Create a message from this agent."""
        self.actions_taken += 1
        return AgentMessage(
            sender=self.name,
            content=content,
            action=action,
            metadata=metadata,
        )
