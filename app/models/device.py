from datetime import datetime, timezone

from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Device(Base):
    __tablename__ = "devices"
    __table_args__ = (
        CheckConstraint("platform IN ('android', 'ios')", name="ck_devices_platform"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    platform: Mapped[str] = mapped_column(String(10), nullable=False)
    manufacturer: Mapped[str | None] = mapped_column(String(100), nullable=True)
    device_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    os_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    call_recording_supported: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    on_device_model_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    embedding_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    push_token: Mapped[str | None] = mapped_column(String(255), nullable=True)
    notification_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    notification_permission: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    microphone_permission: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    file_permission: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    battery_optimization_ignored: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, onupdate=_utc_now, nullable=False
    )
