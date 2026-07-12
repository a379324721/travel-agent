"""审批服务（mock OA）：创建即秒批并保留单据可查，接口留好对接真实 OA。"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(slots=True)
class ApprovalTicket:
    approval_id: str
    employee_id: str
    reason: str
    amount_cny: float
    status: str
    created_at: str
    note: str


class ApprovalService:
    """进程内 mock：真实场景应调用 OA 系统并异步回调审批结果。"""

    def __init__(self) -> None:
        self._tickets: dict[str, ApprovalTicket] = {}

    def create(self, employee_id: str, reason: str, amount_cny: float) -> dict[str, Any]:
        ticket = ApprovalTicket(
            approval_id=f"APV-{uuid.uuid4().hex[:8].upper()}",
            employee_id=employee_id,
            reason=reason,
            amount_cny=float(amount_cny),
            status="approved",
            created_at=datetime.now(UTC).isoformat(),
            note="模拟审批：自动通过（对接真实 OA 后为异步审批）",
        )
        self._tickets[ticket.approval_id] = ticket
        return asdict(ticket)

    def query(self, approval_id: str) -> dict[str, Any] | None:
        ticket = self._tickets.get(approval_id)
        return asdict(ticket) if ticket else None
