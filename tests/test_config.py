"""
PluginConfig 单元测试

测试 Pydantic 配置的默认值和 from_astrbot_config 映射。
"""
import pytest
from astrbot_plugin_sb_qunyou.config import PluginConfig, DebounceConfig


class TestDefaults:
    def test_default_debounce_mode(self):
        cfg = PluginConfig()
        assert cfg.debounce.mode == "time"

    def test_default_topic_threshold(self):
        cfg = PluginConfig()
        assert 0 < cfg.topic.similarity_threshold < 1

    def test_default_emotion_disabled(self):
        cfg = PluginConfig()
        assert cfg.emotion.enabled is True

    def test_default_jargon_min_frequency(self):
        cfg = PluginConfig()
        assert cfg.jargon.min_frequency >= 1

    def test_default_webui_port(self):
        cfg = PluginConfig()
        assert cfg.webui.port > 0


class TestFromAstrbotConfig:
    def test_empty_dict(self):
        cfg = PluginConfig.from_astrbot_config({})
        assert cfg.debounce.mode == "time"  # default

    def test_maps_flat_keys(self):
        raw = {
            "debounce_mode": "off",
            "debounce_time_window_seconds": 5.0,
            "topic_enabled": False,
            "webui_port": 9090,
        }
        cfg = PluginConfig.from_astrbot_config(raw)
        assert cfg.debounce.mode == "off"
        assert cfg.debounce.time_window_seconds == 5.0
        assert cfg.topic.enabled is False
        assert cfg.webui.port == 9090

    def test_partial_config(self):
        raw = {"emotion_sensitivity": 0.8}
        cfg = PluginConfig.from_astrbot_config(raw)
        assert cfg.emotion.sensitivity == 0.8
        assert cfg.debounce.mode == "time"  # unchanged default
