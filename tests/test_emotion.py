"""
EmotionEngine 单元测试

测试：
  - mood → valence/arousal 映射
  - mood 解析
  - sensitivity 概率门逻辑
  - enabled 开关
"""
import pytest
from astrbot_plugin_sb_qunyou.services.emotion import EmotionEngine
from astrbot_plugin_sb_qunyou.config import PluginConfig, EmotionConfig


class TestMoodMapping:
    def setup_method(self):
        cfg = PluginConfig(emotion=EmotionConfig(enabled=True))
        self.engine = EmotionEngine(cfg, llm=None)

    def test_happy_valence_positive(self):
        v = EmotionEngine._mood_to_valence("happy")
        assert v > 0

    def test_angry_valence_negative(self):
        v = EmotionEngine._mood_to_valence("angry")
        assert v < 0

    def test_neutral_valence_zero(self):
        v = EmotionEngine._mood_to_valence("neutral")
        assert v == 0.0

    def test_excited_high_arousal(self):
        a = EmotionEngine._mood_to_arousal("excited")
        assert a > 0.7

    def test_bored_low_arousal(self):
        a = EmotionEngine._mood_to_arousal("bored")
        assert a < 0.3

    def test_unknown_mood_defaults(self):
        v = EmotionEngine._mood_to_valence("unknown")
        a = EmotionEngine._mood_to_arousal("unknown")
        assert v == 0.0
        assert a == 0.3


class TestMoodParsing:
    def setup_method(self):
        cfg = PluginConfig(emotion=EmotionConfig(enabled=True))
        self.engine = EmotionEngine(cfg, llm=None)

    def test_parse_clean(self):
        assert self.engine._parse_mood("happy") == "happy"

    def test_parse_with_noise(self):
        assert self.engine._parse_mood("I think the mood is excited here") == "excited"

    def test_parse_uppercase(self):
        assert self.engine._parse_mood("ANGRY") == "angry"

    def test_parse_no_match(self):
        assert self.engine._parse_mood("unknown_mood") is None

    def test_parse_priority_first_match(self):
        # "happy" appears first in MOODS tuple
        result = self.engine._parse_mood("happy and excited")
        assert result == "happy"
