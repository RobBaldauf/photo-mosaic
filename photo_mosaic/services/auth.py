import jwt
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN

from photo_mosaic.models.app_config import get_config

API_KEY = APIKeyHeader(name="api_key", auto_error=False)
ADMIN_ID = "photo-mosaic-admin"


class AuthService:
    async def admin_auth(self, api_key: str = Security(API_KEY)):
        return await self._auth(api_key, ADMIN_ID)

    @staticmethod
    async def _auth(api_key_header: str, user_id: str) -> str:
        if get_config().enable_auth:
            try:
                decoded_token = jwt.decode(api_key_header, get_config().jwt_secret, algorithms=["HS256"])
                if decoded_token["id"] == user_id:
                    return api_key_header
                raise jwt.DecodeError()
            except jwt.DecodeError as exc:
                raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="api_key header invalid or missing") from exc
        return ""
