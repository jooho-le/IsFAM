from typing import Literal

from pydantic import BaseModel, Field

Platform = Literal["android", "ios"]


class DeviceRegisterRequest(BaseModel):
    platform: Platform = Field(..., examples=["android"])
    manufacturer: str | None = Field(None, examples=["samsung"])
    device_model: str | None = Field(None, examples=["SM-S911N"])
    os_version: str | None = Field(None, examples=["14"])
    push_token: str | None = Field(None, examples=["fcm-token-abc123"])


class DeviceRegisterResponse(BaseModel):
    device_id: int
    call_recording_supported: bool


class DeviceCapabilityResponse(BaseModel):
    call_recording_supported: bool
    guidance_required: bool
    guidance_url: str | None


class DeviceCapabilityUpdateRequest(BaseModel):
    call_recording_supported: bool


class DeviceCapabilityUpdateResponse(BaseModel):
    device_id: int
    call_recording_supported: bool


class PushTokenUpdateRequest(BaseModel):
    push_token: str


class PushTokenUpdateResponse(BaseModel):
    device_id: int
    push_token: str
