import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_token_payload, get_current_user
from app.core.config import Settings, get_settings
from app.core.errors import ApiError
from app.core.security import (
    create_access_token,
    generate_refresh_token_value,
    hash_token,
    mask_phone_number,
)
from app.db.postgres import get_db
from app.models.user import User
from app.repositories.refresh_token_repository import RefreshTokenRepository
from app.repositories.user_repository import UserRepository
from app.schemas.auth import (
    LoginRequest,
    LoginResponse,
    MeResponse,
    RefreshRequest,
    SignupRequest,
    SignupResponse,
    TokenResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


def _issue_tokens(
    *,
    user: User,
    device_id: int | None,
    settings: Settings,
    db: Session,
) -> TokenResponse:
    now = datetime.now(timezone.utc)
    raw_refresh_token = generate_refresh_token_value()
    refresh_token = RefreshTokenRepository(db).create(
        user_id=user.id,
        device_id=device_id,
        token_hash=hash_token(raw_refresh_token),
        issued_at=now,
        expires_at=now + timedelta(days=settings.jwt_refresh_token_expire_days),
    )
    access_token = create_access_token(
        user_id=user.id,
        role=user.role,
        session_id=refresh_token.id,
        settings=settings,
    )
    return TokenResponse(access_token=access_token, refresh_token=raw_refresh_token)


@router.post("/signup", response_model=SignupResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    body: SignupRequest,
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
) -> SignupResponse:
    """회원가입. 데모 정책상 verification_code는 123456으로 고정."""

    if body.verification_code != settings.auth_fixed_verification_code:
        raise ApiError(status.HTTP_422_UNPROCESSABLE_ENTITY, "AUTH_004", "verification code does not match")

    user_repository = UserRepository(db)
    if user_repository.get_by_phone_number(body.phone_number) is not None:
        raise ApiError(status.HTTP_409_CONFLICT, "AUTH_003", "phone number is already registered")

    user = user_repository.create(
        phone_number=body.phone_number,
        display_name=body.display_name,
        role=body.role,
    )
    tokens = _issue_tokens(user=user, device_id=None, settings=settings, db=db)

    return SignupResponse(
        user_id=user.id,
        display_name=user.display_name,
        role=user.role,
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
    )


@router.post("/login", response_model=LoginResponse)
async def login(
    body: LoginRequest,
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
) -> LoginResponse:
    """로그인.

    로그인 흐름: 1) signup 또는 login으로 access_token/refresh_token 발급
    (verification_code는 데모용으로 123456 고정)
    2) 우측 상단 Authorize 버튼에 access_token을 붙여넣으면 이후 인증이 필요한
    엔드포인트에 자동 적용됨
    3) access_token 만료 시 refresh로 재발급 (호출할 때마다 기존 refresh_token은
    폐기되고 새로 발급되는 로테이션 방식)
    4) logout은 현재 세션의 refresh_token만 폐기 (다른 기기 세션엔 영향 없음)
    """

    user = UserRepository(db).get_by_phone_number(body.phone_number)
    if user is None:
        raise ApiError(status.HTTP_401_UNAUTHORIZED, "AUTH_001", "phone number is not registered")

    if body.verification_code != settings.auth_fixed_verification_code:
        raise ApiError(status.HTTP_422_UNPROCESSABLE_ENTITY, "AUTH_004", "verification code does not match")

    tokens = _issue_tokens(user=user, device_id=None, settings=settings, db=db)
    return LoginResponse(
        user_id=user.id,
        role=user.role,
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    payload: dict = Depends(get_current_token_payload),
    db: Session = Depends(get_db),
) -> None:
    """로그아웃. 지금 쓰고 있는 access_token의 세션(refresh_tokens row)만 폐기하며, 다른 기기 로그인 세션에는 영향 없음."""

    session_id = payload.get("sid")
    if session_id is None:
        return

    refresh_token_repository = RefreshTokenRepository(db)
    token = refresh_token_repository.get_by_id(int(session_id))
    if token is not None and token.revoked_at is None:
        refresh_token_repository.revoke(token)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    body: RefreshRequest,
    settings: Settings = Depends(get_settings),
    db: Session = Depends(get_db),
) -> TokenResponse:
    """access_token 재발급. 호출할 때마다 기존 refresh_token은 폐기하고 새 access_token/refresh_token 쌍을 발급 (재사용 방지를 위한 로테이션)."""

    refresh_token_repository = RefreshTokenRepository(db)
    token = refresh_token_repository.get_by_hash(hash_token(body.refresh_token))

    now = datetime.now(timezone.utc)
    if token is None or token.revoked_at is not None or token.expires_at <= now:
        raise ApiError(status.HTTP_401_UNAUTHORIZED, "AUTH_002", "refresh token is expired or invalid")

    user = UserRepository(db).get_by_id(token.user_id)
    if user is None:
        raise ApiError(status.HTTP_401_UNAUTHORIZED, "AUTH_002", "refresh token is expired or invalid")

    refresh_token_repository.revoke(token)
    return _issue_tokens(user=user, device_id=token.device_id, settings=settings, db=db)


@router.get("/me", response_model=MeResponse)
async def get_me(user: User = Depends(get_current_user)) -> MeResponse:
    """내 프로필 조회. phone_number는 마스킹되어(010****1234) 내려감."""

    return MeResponse(
        user_id=user.id,
        display_name=user.display_name,
        phone_number=mask_phone_number(user.phone_number),
        role=user.role,
        created_at=user.created_at,
    )
