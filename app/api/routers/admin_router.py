import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.openapi.models import APIKey
from fastapi.responses import JSONResponse

from app.models.app_config import get_config
from app.models.mosaic_config import MosaicConfig
from app.services.auth import AuthService
from app.services.mosaic_filling import MosaicFillingService
from app.services.mosaic_management import MosaicManagementService
from app.services.sqlite_persistence import SQLiteAbstractPersistenceService
from app.utils.data import validate_request_uuid

router = APIRouter()
auth_service = AuthService()
persistence_service = SQLiteAbstractPersistenceService(get_config().sql_lite_path)
mgmt_service = MosaicManagementService(persistence_service)
filling_service = MosaicFillingService(persistence_service)


@router.post(
    "/mosaic/",
    name="post_mosaic",
    summary="Create a new mosaic with empty segments from an uploaded image and configuration.",
)
async def post_mosaic(
    file: UploadFile = File(...),
    num_segments: int = Form(300),
    mosaic_bg_brightness: float = Form(0.25),
    mosaic_blend_value: float = Form(0.25),
    segment_blend_value: float = Form(0.4),
    segment_blur_low: float = Form(4),
    segment_blur_medium: float = Form(3),
    segment_blur_high: float = Form(2),
    key: APIKey = Depends(auth_service.admin_auth),
) -> JSONResponse:
    # pylint: disable=unused-argument

    image_bytes = await file.read()
    try:
        config = MosaicConfig(
            num_segments=num_segments,
            mosaic_background_brightness=mosaic_bg_brightness,
            mosaic_blend_value=mosaic_blend_value,
            segment_blend_value=segment_blend_value,
            segment_blur_low=segment_blur_low,
            segment_blur_medium=segment_blur_medium,
            segment_blur_high=segment_blur_high,
        )
        mosaic_id = await mgmt_service.create_mosaic(image_bytes, config)
        return JSONResponse(content={"msg": "Mosaic created!"}, headers={"mosaic_id": mosaic_id})
    except HTTPException:
        raise
    except BaseException as exc:
        logging.error("", exc_info=True)
        raise HTTPException(status_code=500, detail="An unknown server error occurred") from exc


@router.post(
    "/mosaic/{mosaic_id}/reset",
    name="reset_mosaic",
    summary="Reset a mosaic to its state after creation. ALL uploaded segment images will be deleted and "
    "can not be restored",
)
async def reset_mosaic(mosaic_id: str, key: APIKey = Depends(auth_service.admin_auth)) -> JSONResponse:
    # pylint: disable=unused-argument
    try:
        m_id = validate_request_uuid(mosaic_id, "mosaic")
        await mgmt_service.reset_mosaic(m_id)
        return JSONResponse(content={"msg": "Mosaic reseted!"}, headers={"mosaic_id": m_id})
    except HTTPException:
        raise
    except BaseException as exc:
        logging.error("", exc_info=True)
        raise HTTPException(status_code=500, detail="An unknown server error occurred") from exc


@router.delete(
    "/mosaic/{mosaic_id}",
    name="delete_mosaic",
    summary="Remove a mosaic completely. All mosaic data and uploaded segment images will be deleted and "
    "can not be restored.",
)
async def delete_mosaic(mosaic_id: str, key: APIKey = Depends(auth_service.admin_auth)) -> JSONResponse:
    # pylint: disable=unused-argument
    try:
        m_id = validate_request_uuid(mosaic_id, "mosaic")
        await mgmt_service.delete_mosaic(m_id)
        return JSONResponse(content={"msg": "Mosaic deleted!"}, headers={"mosaic_id": m_id})
    except HTTPException:
        raise
    except BaseException as exc:
        logging.error("", exc_info=True)
        raise HTTPException(status_code=500, detail="An unknown server error occurred") from exc


@router.post(
    "/mosaic/{mosaic_id}/segment",
    name="post_mosaic_segment_random",
    summary="Extension of the '/mosaic/{mosaic_id}/segment/{segment_id}' endpoint to random sampling and quick filling "
    "for testing purposes.",
)
async def post_mosaic_segment_random(
    mosaic_id: str,
    quick_fill: bool = Form(True),
    file: UploadFile = File(...),
    key: APIKey = Depends(auth_service.public_auth),
) -> JSONResponse:
    # pylint: disable=unused-argument
    try:
        input_image_bytes = await file.read()
        m_id = validate_request_uuid(mosaic_id, "mosaic")
        await filling_service.fill_random_segment(m_id, input_image_bytes, quick_fill)
        return JSONResponse(content={"msg": "Segment filled!"}, headers={"mosaic_id": m_id})
    except HTTPException:
        raise
    except BaseException as exc:
        logging.error("", exc_info=True)
        raise HTTPException(status_code=500, detail="An unknown server error occurred") from exc
