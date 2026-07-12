"""SQLAlchemy 表模型：订票与报销记录。"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utc_now() -> datetime:
    return datetime.now(UTC)


class BookingRow(Base):
    __tablename__ = "bookings"

    booking_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    employee_id: Mapped[str] = mapped_column(String(64), index=True)
    booking_type: Mapped[str] = mapped_column(String(16))
    inventory_id: Mapped[str] = mapped_column(String(128))
    confirmation_code: Mapped[str] = mapped_column(String(32))
    amount_cny: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(16), default="confirmed")
    reimbursement_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)


class ReimbursementRow(Base):
    __tablename__ = "reimbursements"

    reimbursement_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    employee_id: Mapped[str] = mapped_column(String(64), index=True)
    total_cny: Mapped[float] = mapped_column(Float, default=0.0)
    booking_count: Mapped[int] = mapped_column(default=0)
    status: Mapped[str] = mapped_column(String(16), default="submitted")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now)
