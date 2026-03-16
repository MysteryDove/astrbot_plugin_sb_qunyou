"""
JargonService 单元测试

测试：
  - jieba 分词与计数
  - 标点/数字过滤
  - JSON 解析 (批量推断结果)
  - 计数器隔离 (不同 group)
"""
import pytest
from astrbot_plugin_sb_qunyou.config import PluginConfig, JargonConfig
from astrbot_plugin_sb_qunyou.services.jargon import JargonService


def make_jargon(enabled: bool = True) -> JargonService:
    cfg = PluginConfig(jargon=JargonConfig(enabled=enabled, min_frequency=3))
    return JargonService(cfg, llm=None)


class TestWordCounting:
    def test_basic_counting(self):
        svc = make_jargon()
        svc.count_words("g1", "今天天气真好天气不错")
        counter = svc._counters.get("g1")
        assert counter is not None
        # "天气" should appear 2x
        assert counter["天气"] == 2

    def test_short_tokens_filtered(self):
        svc = make_jargon()
        svc.count_words("g1", "我在吃饭")
        counter = svc._counters.get("g1")
        # Single-char tokens like "我", "在" should be filtered
        for token in counter:
            assert len(token) >= 2

    def test_digits_filtered(self):
        svc = make_jargon()
        svc.count_words("g1", "服务器 123 端口 456")
        counter = svc._counters.get("g1")
        assert "123" not in counter
        assert "456" not in counter

    def test_punctuation_filtered(self):
        svc = make_jargon()
        svc.count_words("g1", "你好！这个很棒，真的。")
        counter = svc._counters.get("g1")
        for token in counter:
            assert token not in "，。！？、；：""''（）【】…"

    def test_groups_independent(self):
        svc = make_jargon()
        svc.count_words("g1", "游戏游戏")
        svc.count_words("g2", "音乐音乐音乐")
        assert "g1" in svc._counters
        assert "g2" in svc._counters
        assert svc._counters["g1"]["游戏"] == 2
        assert svc._counters["g2"]["音乐"] == 3

    def test_disabled_no_counting(self):
        svc = make_jargon(enabled=False)
        svc.count_words("g1", "今天天气真好")
        assert "g1" not in svc._counters


class TestMeaningParsing:
    def setup_method(self):
        self.svc = make_jargon()

    def test_parse_clean_json(self):
        response = '{"yyds": "永远的神", "awsl": "啊我死了"}'
        result = self.svc._parse_meanings(response)
        assert result == {"yyds": "永远的神", "awsl": "啊我死了"}

    def test_parse_code_block(self):
        response = '```json\n{"test": "value"}\n```'
        result = self.svc._parse_meanings(response)
        assert result == {"test": "value"}

    def test_parse_empty_response(self):
        result = self.svc._parse_meanings("")
        assert result == {}

    def test_parse_invalid_json(self):
        result = self.svc._parse_meanings("not json at all")
        assert result == {}

    def test_parse_array_not_dict(self):
        result = self.svc._parse_meanings('["not", "a", "dict"]')
        assert result == {}
