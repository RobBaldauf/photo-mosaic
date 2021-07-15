import uuid

from fastapi import HTTPException


def is_valid_uuid(uuid_string: str) -> bool:
    try:
        uuid.UUID(uuid_string)
        return True
    except ValueError:
        return False


def validate_request_uuid(uuid_string: str, id_label: str) -> str:
    stripped_uuid = str(uuid_string).strip()
    if is_valid_uuid(stripped_uuid):
        return stripped_uuid
    raise HTTPException(
        status_code=400,
        detail=f"Invalid {id_label} id: {stripped_uuid}!",
    )
