import uuid

from fastapi import HTTPException

from photo_mosaic.services.persistence import db


def validate_request_uuid(uuid_string: str, id_label: str) -> str:
    stripped_uuid = str(uuid_string).strip()
    if not uuid_format_is_valid(stripped_uuid):
        raise HTTPException(
            status_code=400,
            detail=f"{id_label} id ({uuid_string}) has to be of format UUID4!",
        )
    if not uuid_exists(stripped_uuid, id_label):
        raise HTTPException(status_code=404, detail=f"{id_label} id ({uuid_string}) does not exist.")
    return stripped_uuid


def uuid_format_is_valid(uuid_string: str) -> bool:
    try:
        uuid.UUID(uuid_string)
        return True
    except ValueError:
        return False


def uuid_exists(uuid_string: str, id_label: str) -> bool:
    if id_label == "mosaic":
        return db.mosaic_exists(uuid_string)
    return db.segment_exists(uuid_string)


def generate_id() -> str:
    return str(uuid.uuid4())
