from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.device import Device


class DeviceRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, device_id: int) -> Device | None:
        return self.db.get(Device, device_id)

    def get_by_user_and_push_token(self, user_id: int, push_token: str) -> Device | None:
        return self.db.scalar(
            select(Device).where(Device.user_id == user_id, Device.push_token == push_token)
        )

    def create(
        self,
        *,
        user_id: int,
        platform: str,
        manufacturer: str | None,
        device_model: str | None,
        os_version: str | None,
        push_token: str | None,
        call_recording_supported: bool,
    ) -> Device:
        device = Device(
            user_id=user_id,
            platform=platform,
            manufacturer=manufacturer,
            device_model=device_model,
            os_version=os_version,
            push_token=push_token,
            call_recording_supported=call_recording_supported,
        )
        self.db.add(device)
        self.db.commit()
        self.db.refresh(device)
        return device

    def update_registration_info(
        self,
        device: Device,
        *,
        platform: str,
        manufacturer: str | None,
        device_model: str | None,
        os_version: str | None,
    ) -> Device:
        # call_recording_supported is left untouched: it may already reflect a
        # client-verified result from PUT /devices/me/capability.
        device.platform = platform
        device.manufacturer = manufacturer
        device.device_model = device_model
        device.os_version = os_version
        self.db.add(device)
        self.db.commit()
        self.db.refresh(device)
        return device

    def update_capability(self, device: Device, *, call_recording_supported: bool) -> Device:
        device.call_recording_supported = call_recording_supported
        self.db.add(device)
        self.db.commit()
        self.db.refresh(device)
        return device

    def update_push_token(self, device: Device, *, push_token: str) -> Device:
        device.push_token = push_token
        self.db.add(device)
        self.db.commit()
        self.db.refresh(device)
        return device
