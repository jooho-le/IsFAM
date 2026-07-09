from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["child", "parent"]


class SignupRequest(BaseModel):
    phone_number: str = Field(..., examples=["01012341234"])
    display_name: str = Field(..., examples=["홍길동"])
    verification_code: str = Field(..., examples=["123456"])
    role: Role = Field(
        ...,
        description="child(음성을 등록하는 자녀 계정) 또는 parent(보호받는 부모 계정) 중 하나",
        examples=["parent"],
    )


class LoginRequest(BaseModel):
    phone_number: str = Field(..., examples=["01012341234"])
    verification_code: str = Field(..., examples=["123456"])


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str


class SignupResponse(TokenResponse):
    user_id: int
    display_name: str
    role: Role


class LoginResponse(TokenResponse):
    user_id: int
    role: Role


class MeResponse(BaseModel):
    user_id: int
    display_name: str
    phone_number: str
    role: Role
    created_at: datetime
