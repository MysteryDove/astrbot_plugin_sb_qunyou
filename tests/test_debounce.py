"""
DebounceManager 单元测试 — 事件驱动架构

由于事件驱动防抖需要真实的 AstrMessageEvent (包含 stop_event, unified_msg_origin
等方法)，我们只能测试子组件的纯逻辑：
  - MessageParser.is_command
  - MessageParser.parse_message (mock message_obj)
  - ForwardHandler._parse_raw
  - DebounceSession 数据结构
  - L2 语义完整性 prompt 存在性验证
"""
import pytest
from astrbot_plugin_sb_qunyou.pipeline.debounce import (
    DebounceSession,
    MessageParser,
    ForwardHandler,
)


# ------------------------------------------------------------------ #
#  MessageParser.is_command
# ------------------------------------------------------------------ #

class TestIsCommand:
    def test_basic_command(self):
        assert MessageParser.is_command("/help", ["/", "#"]) is True

    def test_hash_command(self):
        assert MessageParser.is_command("#image test", ["/", "#"]) is True

    def test_not_command(self):
        assert MessageParser.is_command("hello world", ["/", "#"]) is False

    def test_empty_text(self):
        assert MessageParser.is_command("", ["/", "#"]) is False

    def test_whitespace_prefix(self):
        assert MessageParser.is_command("  /help", ["/", "#"]) is True

    def test_custom_prefixes(self):
        assert MessageParser.is_command("!ban user", ["!"]) is True
        assert MessageParser.is_command("hello", ["!"]) is False


# ------------------------------------------------------------------ #
#  ForwardHandler._parse_raw
# ------------------------------------------------------------------ #

class TestParseRaw:
    def test_list_input(self):
        raw = [{"type": "text", "data": {"text": "hello"}}]
        assert ForwardHandler._parse_raw(raw) == raw

    def test_json_string(self):
        import json
        raw = json.dumps([{"type": "text", "data": {"text": "hi"}}])
        result = ForwardHandler._parse_raw(raw)
        assert len(result) == 1
        assert result[0]["type"] == "text"

    def test_plain_string(self):
        result = ForwardHandler._parse_raw("hello world")
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["data"]["text"] == "hello world"

    def test_other_type(self):
        assert ForwardHandler._parse_raw(42) == []

    def test_invalid_json(self):
        result = ForwardHandler._parse_raw("{not valid json")
        assert len(result) == 1  # treated as plain string


# ------------------------------------------------------------------ #
#  DebounceSession dataclass
# ------------------------------------------------------------------ #

class TestDebounceSession:
    def test_defaults(self):
        session = DebounceSession()
        assert session.buffer == []
        assert session.images == []
        assert session.is_typing is False
        assert session.timer_task is None
        assert not session.flush_event.is_set()

    def test_buffer_accumulation(self):
        session = DebounceSession()
        session.buffer.append("msg1")
        session.buffer.append("msg2")
        assert len(session.buffer) == 2

    def test_images_accumulation(self):
        session = DebounceSession()
        session.images.append("http://img.png")
        assert len(session.images) == 1

    def test_typing_state(self):
        session = DebounceSession()
        session.is_typing = True
        assert session.is_typing is True


# ------------------------------------------------------------------ #
#  L2 semantic prompt existence
# ------------------------------------------------------------------ #

class TestSemanticPrompt:
    def test_prompt_exists(self):
        from astrbot_plugin_sb_qunyou.prompts.templates import SEMANTIC_COMPLETENESS
        assert "{text}" in SEMANTIC_COMPLETENESS
        assert "complete" in SEMANTIC_COMPLETENESS.lower()
        assert "incomplete" in SEMANTIC_COMPLETENESS.lower()
