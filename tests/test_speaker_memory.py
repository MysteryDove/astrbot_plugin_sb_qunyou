"""
SpeakerMemoryService 单元测试

测试 LLM 事实抽取的 JSON 解析逻辑 (不需要真正的 LLM)。
"""
import pytest
from astrbot_plugin_sb_qunyou.config import PluginConfig
from astrbot_plugin_sb_qunyou.services.speaker_memory import SpeakerMemoryService


def make_speaker_memory() -> SpeakerMemoryService:
    cfg = PluginConfig()
    return SpeakerMemoryService(cfg, llm=None)


class TestFactParsing:
    def setup_method(self):
        self.svc = make_speaker_memory()

    def test_parse_clean_array(self):
        resp = '["喜欢打游戏", "是大学生"]'
        facts = self.svc._parse_facts(resp)
        assert facts == ["喜欢打游戏", "是大学生"]

    def test_parse_code_block(self):
        resp = '```json\n["fact1", "fact2"]\n```'
        facts = self.svc._parse_facts(resp)
        assert facts == ["fact1", "fact2"]

    def test_parse_empty_array(self):
        resp = "[]"
        facts = self.svc._parse_facts(resp)
        assert facts == []

    def test_parse_invalid_json(self):
        resp = "这不是 JSON"
        facts = self.svc._parse_facts(resp)
        assert facts == []

    def test_parse_dict_not_array(self):
        resp = '{"key": "value"}'
        facts = self.svc._parse_facts(resp)
        assert facts == []

    def test_parse_filters_empty_strings(self):
        resp = '["", "real fact", ""]'
        facts = self.svc._parse_facts(resp)
        assert facts == ["real fact"]

    def test_parse_mixed_types(self):
        resp = '["text", 42, true, null]'
        facts = self.svc._parse_facts(resp)
        # Should convert non-strings to str, filter None
        assert "text" in facts
        assert "42" in facts

    def test_parse_code_block_without_json_label(self):
        resp = '```\n["a", "b"]\n```'
        facts = self.svc._parse_facts(resp)
        assert facts == ["a", "b"]
