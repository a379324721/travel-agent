"""商旅子域意图定义（供规则引擎、LLM 分类器、识别器共同引用）。"""

from __future__ import annotations

from enum import Enum


class TravelIntent(str, Enum):
    """value 即 slug，description 供慢车道提示词使用：
    slug 命名不自解释（如 application、rag），不给含义 LLM 只能靠猜，
    边界类（policy vs rag、search vs booking）会随机站队。"""

    description: str

    def __new__(cls, slug: str, description: str) -> TravelIntent:
        obj = str.__new__(cls, slug)
        obj._value_ = slug
        obj.description = description
        return obj

    SEARCH_FLIGHT = "search_flight", "查询/搜索机票、航班信息"
    SEARCH_HOTEL = "search_hotel", "查询/搜索酒店、住宿"
    SEARCH_TRAIN = "search_train", "查询/搜索火车票、高铁、车次"
    TRIP_PLANNING = "trip_planning", "规划、安排差旅行程"
    APPLICATION = "application", "提交出差申请、审批、报备或报销单"
    POLICY = "policy", "咨询差旅标准、额度、合规、审批线等公司政策"
    BOOKING = "booking", "下单预订机票、酒店、火车票"
    INFO_QUERY = "info_query", "查询航班动态、天气、价格、时间等一般信息"
    RAG = "rag", "需要引用公司内部规定、制度、手册原文回答的问题"
    GENERAL = "general", "闲聊或无法归入以上类别"
