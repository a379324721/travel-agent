"""订票与报销记录存储：PostgreSQL 可用时落库，否则进程内 mock。"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.infrastructure.database.models import BookingRow, ReimbursementRow
from app.infrastructure.database.session import session_scope


def _new_reimbursement_id() -> str:
    return f"RMB-{uuid.uuid4().hex[:8].upper()}"


class InMemoryBookingStore:
    """无 PostgreSQL 时的进程内实现，接口与 PgBookingStore 一致。"""

    def __init__(self) -> None:
        self._bookings: list[dict[str, Any]] = []
        self._reimbursements: list[dict[str, Any]] = []

    async def add_booking(
        self,
        *,
        booking_id: str,
        employee_id: str,
        booking_type: str,
        inventory_id: str,
        confirmation_code: str,
        amount_cny: float,
    ) -> dict[str, Any]:
        row = {
            "booking_id": booking_id,
            "employee_id": employee_id,
            "booking_type": booking_type,
            "inventory_id": inventory_id,
            "confirmation_code": confirmation_code,
            "amount_cny": float(amount_cny),
            "status": "confirmed",
            "reimbursement_id": None,
            "created_at": datetime.now(UTC).isoformat(),
        }
        self._bookings.append(row)
        return dict(row)

    async def list_unreimbursed(self, employee_id: str) -> list[dict[str, Any]]:
        return [
            dict(b)
            for b in self._bookings
            if b["employee_id"] == employee_id and b["reimbursement_id"] is None
        ]

    async def create_reimbursement(self, employee_id: str) -> dict[str, Any]:
        pending = [
            b
            for b in self._bookings
            if b["employee_id"] == employee_id and b["reimbursement_id"] is None
        ]
        if not pending:
            return {"ok": False, "error": "该员工没有可报销的订单。"}
        rid = _new_reimbursement_id()
        total = round(sum(b["amount_cny"] for b in pending), 2)
        for b in pending:
            b["reimbursement_id"] = rid
        record = {
            "reimbursement_id": rid,
            "employee_id": employee_id,
            "total_cny": total,
            "booking_count": len(pending),
            "status": "submitted",
            "created_at": datetime.now(UTC).isoformat(),
        }
        self._reimbursements.append(record)
        return {"ok": True, **record, "bookings": [dict(b) for b in pending]}


class PgBookingStore:
    """PostgreSQL 实现：订票与报销单落库。"""

    def __init__(self, factory: async_sessionmaker[AsyncSession]) -> None:
        self._factory = factory

    @staticmethod
    def _booking_dict(row: BookingRow) -> dict[str, Any]:
        return {
            "booking_id": row.booking_id,
            "employee_id": row.employee_id,
            "booking_type": row.booking_type,
            "inventory_id": row.inventory_id,
            "confirmation_code": row.confirmation_code,
            "amount_cny": row.amount_cny,
            "status": row.status,
            "reimbursement_id": row.reimbursement_id,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }

    async def add_booking(
        self,
        *,
        booking_id: str,
        employee_id: str,
        booking_type: str,
        inventory_id: str,
        confirmation_code: str,
        amount_cny: float,
    ) -> dict[str, Any]:
        async with session_scope(self._factory) as session:
            row = BookingRow(
                booking_id=booking_id,
                employee_id=employee_id,
                booking_type=booking_type,
                inventory_id=inventory_id,
                confirmation_code=confirmation_code,
                amount_cny=float(amount_cny),
            )
            session.add(row)
            await session.flush()
            return self._booking_dict(row)

    async def list_unreimbursed(self, employee_id: str) -> list[dict[str, Any]]:
        async with session_scope(self._factory) as session:
            rows = (
                await session.execute(
                    select(BookingRow).where(
                        BookingRow.employee_id == employee_id,
                        BookingRow.reimbursement_id.is_(None),
                    )
                )
            ).scalars()
            return [self._booking_dict(r) for r in rows]

    async def create_reimbursement(self, employee_id: str) -> dict[str, Any]:
        async with session_scope(self._factory) as session:
            rows = list(
                (
                    await session.execute(
                        select(BookingRow).where(
                            BookingRow.employee_id == employee_id,
                            BookingRow.reimbursement_id.is_(None),
                        )
                    )
                ).scalars()
            )
            if not rows:
                return {"ok": False, "error": "该员工没有可报销的订单。"}
            rid = _new_reimbursement_id()
            total = round(sum(r.amount_cny for r in rows), 2)
            for r in rows:
                r.reimbursement_id = rid
            record = ReimbursementRow(
                reimbursement_id=rid,
                employee_id=employee_id,
                total_cny=total,
                booking_count=len(rows),
            )
            session.add(record)
            await session.flush()
            return {
                "ok": True,
                "reimbursement_id": rid,
                "employee_id": employee_id,
                "total_cny": total,
                "booking_count": len(rows),
                "status": record.status,
                "created_at": record.created_at.isoformat() if record.created_at else None,
                "bookings": [self._booking_dict(r) for r in rows],
            }
