from typing import List

import jwt
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN

from app.models.app_config import get_config

API_KEY_HEADER_PUBLIC = APIKeyHeader(name="api_key", auto_error=False)
API_KEY_HEADER_ADMIN = APIKeyHeader(name="api_key", auto_error=False)
PUBLIC_ID = "public"
ADMIN_ID = "admin"


class AuthService:
    async def public_auth(
        self,
        api_key_header: str = Security(API_KEY_HEADER_PUBLIC),
    ):
        return await self._auth(api_key_header, [PUBLIC_ID, ADMIN_ID])

    async def admin_auth(
        self,
        api_key_header: str = Security(API_KEY_HEADER_PUBLIC),
    ):
        return await self._auth(api_key_header, [ADMIN_ID])

    @staticmethod
    async def _auth(api_key_header: str, user_ids: List[str]) -> str:
        if get_config().enable_auth:
            try:
                decoded_token = jwt.decode(api_key_header, get_config().jwt_secret, algorithms=["HS256"])
                if decoded_token["id"] in user_ids:
                    return api_key_header
                raise jwt.DecodeError()
            except jwt.DecodeError as exc:
                raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="API_KEY invalid or missing") from exc
        return ""
