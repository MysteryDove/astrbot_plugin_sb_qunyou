"""
ContextBuilder 单元测试

测试注入文本格式化 — 这些是纯函数，不需要 DB 或 LLM mock。
"""
import pytest
from astrbot_plugin_sb_qunyou.pipeline.context_builder import ContextBuilder


class TestPersonaInjection:
    def test_non_empty(self):
        result = ContextBuilder.build_persona_injection("这是一个游戏群")
        assert "<group_persona>" in result
        assert "这是一个游戏群" in result
        assert "</group_persona>" in result

    def test_empty_returns_empty(self):
        assert ContextBuilder.build_persona_injection("") == ""


class TestEmotionInjection:
    def test_non_neutral(self):
        result = ContextBuilder.build_emotion_injection("happy")
        assert "<emotion_state>" in result
        assert "happy" in result

    def test_neutral_returns_empty(self):
        assert ContextBuilder.build_emotion_injection("neutral") == ""

    def test_empty_returns_empty(self):
        assert ContextBuilder.build_emotion_injection("") == ""


class TestThreadInjection:
    def test_with_messages(self):
        msgs = [
            {"sender": "Alice", "text": "hello"},
            {"sender": "Bob", "text": "hi there"},
        ]
        result = ContextBuilder.build_thread_injection("游戏讨论", msgs)
        assert "<thread_context" in result
        assert "游戏讨论" in result
        assert "[Alice]: hello" in result
        assert "[Bob]: hi there" in result

    def test_empty_messages(self):
        assert ContextBuilder.build_thread_injection("topic", []) == ""

    def test_no_topic_defaults(self):
        result = ContextBuilder.build_thread_injection("", [{"sender": "A", "text": "x"}])
        assert 'topic="ongoing"' in result


class TestMemoryInjection:
    def test_with_facts(self):
        facts = ["喜欢原神", "是大学生"]
        result = ContextBuilder.build_memory_injection("user123", facts)
        assert "<user_memories" in result
        assert "user123" in result
        assert "- 喜欢原神" in result
        assert "- 是大学生" in result

    def test_empty_facts(self):
        assert ContextBuilder.build_memory_injection("u1", []) == ""


class TestJargonInjection:
    def test_with_matches(self):
        matches = [("yyds", "永远的神"), ("awsl", "啊我死了")]
        result = ContextBuilder.build_jargon_injection(matches)
        assert "<jargon_hints" in result
        assert "「yyds」= 永远的神" in result
        assert "「awsl」= 啊我死了" in result

    def test_empty_matches(self):
        assert ContextBuilder.build_jargon_injection([]) == ""
