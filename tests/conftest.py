"""测试全局配置。"""

import pytest

from app.config import settings


@pytest.fixture(autouse=True)
def _disable_intent_slow_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    """关闭意图识别慢车道：FakeLLM 的应答队列是给主流程准备的，
    不能被意图分类调用消耗。慢车道逻辑在 test_p2 中用独立的分类器单测覆盖。"""
    monkeypatch.setattr(settings, "intent_slow_lane_enabled", False)
