from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_role
from app.db.postgres import get_db
from app.models.device import Device
from app.models.user import User
from app.repositories.device_repository import DeviceRepository
from app.schemas.device import (
    DeviceCapabilityResponse,
    DeviceCapabilityUpdateRequest,
    DeviceCapabilityUpdateResponse,
    DeviceRegisterRequest,
    DeviceRegisterResponse,
    PushTokenUpdateRequest,
    PushTokenUpdateResponse,
)

router = APIRouter(prefix="/devices", tags=["device"])

_SAMSUNG_CALL_RECORDING_GUIDE_URL = "https://isfam.app/guide/samsung-call-recording"


def _is_samsung(manufacturer: str | None) -> bool:
    return manufacturer is not None and "samsung" in manufacturer.lower()


def _guidance_url_for(manufacturer: str | None) -> str | None:
    # Only Samsung ships a documented call-recording setup flow today; other
    # manufacturers fall back to the client's "unsupported device" messaging.
    return _SAMSUNG_CALL_RECORDING_GUIDE_URL if _is_samsung(manufacturer) else None


def _get_owned_device(device_id: int, user: User, db: Session) -> Device:
    device = DeviceRepository(db).get_by_id(device_id)
    if device is None or device.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "device not found")
    return device


@router.post("", response_model=DeviceRegisterResponse, status_code=status.HTTP_201_CREATED)
async def register_device(
    body: DeviceRegisterRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DeviceRegisterResponse:
    """앱 최초 실행/로그인 시 단말 정보 등록.

    - 호출 조건: 인증만 필요 (role 제한 없음, 본인 계정으로 등록됨)
    - call_recording_supported 초기값: manufacturer에 "samsung"이 포함되면 true, 그 외 false
      (추정치일 뿐이며 실제 값은 온보딩 확인 후 PUT /devices/me/capability로 갱신)
    - push_token은 앱 설치당 고유하므로, 같은 (user, push_token) 조합이 이미 등록돼 있으면
      새로 만들지 않고 기존 row를 갱신 (재실행/네트워크 재시도로 인한 중복 등록 방지)
    """

    device_repository = DeviceRepository(db)
    existing_device = (
        device_repository.get_by_user_and_push_token(user.id, body.push_token)
        if body.push_token
        else None
    )

    if existing_device is not None:
        device = device_repository.update_registration_info(
            existing_device,
            platform=body.platform,
            manufacturer=body.manufacturer,
            device_model=body.device_model,
            os_version=body.os_version,
        )
    else:
        device = device_repository.create(
            user_id=user.id,
            platform=body.platform,
            manufacturer=body.manufacturer,
            device_model=body.device_model,
            os_version=body.os_version,
            push_token=body.push_token,
            call_recording_supported=_is_samsung(body.manufacturer),
        )

    return DeviceRegisterResponse(
        device_id=device.id,
        call_recording_supported=device.call_recording_supported,
    )


@router.get("/me/capability", response_model=DeviceCapabilityResponse)
async def get_capability(
    x_device_id: int = Header(..., alias="X-Device-Id"),
    user: User = Depends(require_role("parent")),
    db: Session = Depends(get_db),
) -> DeviceCapabilityResponse:
    """단말의 통화 자동녹음 + 온디바이스 분석 지원 여부 조회.

    - 호출 조건: role=parent 계정 + X-Device-Id가 본인 소유 device여야 함 (아니면 403/404)
    - guidance_url은 manufacturer에 "samsung"이 포함된 기기에서만 채워짐, 그 외 제조사는 null
      (클라이언트는 null이면 "이 기기에서는 지원하지 않음" 안내로 분기)
    """

    device = _get_owned_device(x_device_id, user, db)
    return DeviceCapabilityResponse(
        call_recording_supported=device.call_recording_supported,
        guidance_required=not device.call_recording_supported,
        guidance_url=_guidance_url_for(device.manufacturer),
    )


@router.put("/me/capability", response_model=DeviceCapabilityUpdateResponse)
async def update_capability(
    body: DeviceCapabilityUpdateRequest,
    x_device_id: int = Header(..., alias="X-Device-Id"),
    user: User = Depends(require_role("parent")),
    db: Session = Depends(get_db),
) -> DeviceCapabilityUpdateResponse:
    """설정 앱 딥링크 복귀 후, 클라이언트가 자체 확인한 통화녹음 지원 결과를 서버에 반영.

    클라이언트 확인 방식 예시: 사용자가 통화녹음 설정을 켠 뒤, 앱이 실제 테스트 통화를 하거나
    삼성 통화녹음 저장 폴더(예: /Call/에 녹음 파일이 실제로 생성되는지)를 스캔해서
    call_recording_supported 값을 판단한 뒤 이 API로 전달.

    호출 조건: role=parent 계정 + X-Device-Id가 본인 소유 device여야 함 (아니면 403/404)."""

    device = _get_owned_device(x_device_id, user, db)
    device_repository = DeviceRepository(db)
    device = device_repository.update_capability(
        device, call_recording_supported=body.call_recording_supported
    )
    return DeviceCapabilityUpdateResponse(
        device_id=device.id,
        call_recording_supported=device.call_recording_supported,
    )


@router.put("/me/push-token", response_model=PushTokenUpdateResponse)
async def update_push_token(
    body: PushTokenUpdateRequest,
    x_device_id: int = Header(..., alias="X-Device-Id"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PushTokenUpdateResponse:
    """푸시 토큰 재발급 시 갱신.

    호출 조건: 인증 필요 (role 제한 없음) + X-Device-Id가 본인 소유 device여야 함 (아니면 404)."""

    device = _get_owned_device(x_device_id, user, db)
    device_repository = DeviceRepository(db)
    device = device_repository.update_push_token(device, push_token=body.push_token)
    return PushTokenUpdateResponse(device_id=device.id, push_token=device.push_token)
