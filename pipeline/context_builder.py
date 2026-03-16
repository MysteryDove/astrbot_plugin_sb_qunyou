"""
上下文构建器 — 将各数据源组装为注入文本

HookHandler 调用此模块来格式化注入内容。
测试时可以直接测试这个模块而不需要 mock AstrBot event。
"""
from __future__ import annotations

from ..prompts.templates import (
    INJECTION_EMOTION,
    INJECTION_GROUP_PERSONA,
    INJECTION_JARGON,
    INJECTION_THREAD_CONTEXT,
    INJECTION_USER_MEMORIES,
)


class ContextBuilder:
    """Assembles context injection strings from raw data."""

    @staticmethod
    def build_persona_injection(persona_prompt: str) -> str:
        """Format group persona for system_prompt injection."""
        if not persona_prompt:
            return ""
        return INJECTION_GROUP_PERSONA.format(persona=persona_prompt)

    @staticmethod
    def build_emotion_injection(mood: str) -> str:
        """Format emotion state for system_prompt injection."""
        if not mood or mood == "neutral":
            return ""
        return INJECTION_EMOTION.format(mood=mood)

    @staticmethod
    def build_thread_injection(
        topic: str, messages: list[dict]
    ) -> str:
        """Format thread context for extra content injection."""
        if not messages:
            return ""
        lines = []
        for m in messages:
            sender = m.get("sender", "?")
            text = m.get("text", "")
            lines.append(f"[{sender}]: {text}")
        return INJECTION_THREAD_CONTEXT.format(
            topic=topic or "ongoing",
            messages="\n".join(lines),
        )

    @staticmethod
    def build_memory_injection(user_id: str, facts: list[str]) -> str:
        """Format user memories for extra content injection."""
        if not facts:
            return ""
        return INJECTION_USER_MEMORIES.format(
            user_id=user_id,
            memories="\n".join(f"- {f}" for f in facts),
        )

    @staticmethod
    def build_jargon_injection(matches: list[tuple[str, str]]) -> str:
        """Format jargon hints for extra content injection."""
        if not matches:
            return ""
        hints = "\n".join(f"「{t}」= {m}" for t, m in matches)
        return INJECTION_JARGON.format(hints=hints)
