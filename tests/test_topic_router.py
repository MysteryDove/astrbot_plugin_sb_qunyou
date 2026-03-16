"""
TopicThreadRouter 单元测试

测试话题路由的核心逻辑：
  - cosine similarity 计算
  - sliding average centroid 更新
  - 路由开关
"""
import pytest
from astrbot_plugin_sb_qunyou.pipeline.topic_router import _cosine_sim, _sliding_average


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        sim = _cosine_sim(v, v)
        assert abs(sim - 1.0) < 1e-5

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        sim = _cosine_sim(a, b)
        assert abs(sim) < 1e-5

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        sim = _cosine_sim(a, b)
        assert abs(sim - (-1.0)) < 1e-5

    def test_known_similarity(self):
        a = [1.0, 1.0]
        b = [1.0, 0.0]
        # cos(45°) ≈ 0.7071
        sim = _cosine_sim(a, b)
        assert abs(sim - 0.7071) < 0.01

    def test_zero_vector(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        sim = _cosine_sim(a, b)
        assert sim == 0.0


class TestSlidingAverage:
    def test_first_message_dominates(self):
        old = [1.0, 0.0]
        new = [0.0, 1.0]
        # count=0 → alpha=1.0, result should be new
        result = _sliding_average(old, new, count=0)
        assert abs(result[0] - 0.0) < 1e-5
        assert abs(result[1] - 1.0) < 1e-5

    def test_many_messages_stable(self):
        old = [1.0, 0.0]
        new = [0.0, 1.0]
        # count=99 → alpha=0.01, result should be close to old
        result = _sliding_average(old, new, count=99)
        assert result[0] > 0.98
        assert result[1] < 0.02

    def test_equal_weight(self):
        old = [2.0, 0.0]
        new = [0.0, 2.0]
        # count=1 → alpha=0.5
        result = _sliding_average(old, new, count=1)
        assert abs(result[0] - 1.0) < 1e-5
        assert abs(result[1] - 1.0) < 1e-5
