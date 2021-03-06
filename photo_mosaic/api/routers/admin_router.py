import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.openapi.models import APIKey
from fastapi.responses import JSONResponse

from photo_mosaic.models.mosaic_config import MosaicConfig
from photo_mosaic.services.auth import AuthService
from photo_mosaic.services.mosaic_filling import filling_service
from photo_mosaic.services.mosaic_management import mgmt_service
from photo_mosaic.utils.request_validation import validate_request_uuid

router = APIRouter()
auth_service = AuthService()


@router.post(
    "/mosaic/",
    name="post_mosaic",
    summary="Create a new mosaic with empty segments from an uploaded image and configuration.",
)
async def post_mosaic(
    file: UploadFile = File(..., description="The image that shall be transformed into a mosaic."),
    title: str = Form(
        "",
        description="The title of the mosaic image",
    ),
    num_segments: int = Form(
        300,
        ge=10,
        le=10000,
        description="The desired number of segments in which the mosaic shall be structured. "
        "The actual number might differ since the API will try to find an optimal "
        "row/column ratio that minimizes the area of unused pixels.",
    ),
    mosaic_bg_brightness: float = Form(
        0.25, ge=0.01, le=1, description="Factor for the brightness reduction of the mosaic background."
    ),
    mosaic_blend_value: float = Form(
        0.25,
        ge=0.01,
        le=1,
        description="Blend factor for merging uploaded portraits with mosaic segments "
        "when adding them to the mosaic. (0.5=equal split)",
    ),
    segment_blend_value: float = Form(
        0.4,
        ge=0.01,
        le=1,
        description="Blend factor for merging uploaded portraits with mosaic segments"
        " when creating a style filter for the user (0.5=equal split)",
    ),
    segment_blur_low: float = Form(
        4,
        ge=0,
        le=5,
        description="The radius of the blur applied to low brightness segments when"
        " creating a style filter for the user.",
    ),
    segment_blur_medium: float = Form(
        3,
        ge=0,
        le=5,
        description="The radius of the blur applied to medium brightness segments when"
        " creating a style filter for the user.",
    ),
    segment_blur_high: float = Form(
        2,
        ge=0,
        le=5,
        description="The radius of the blur applied to high brightness segments when"
        " creating a style filter for the user.",
    ),
    key: APIKey = Depends(auth_service.admin_auth),
) -> JSONResponse:
    # pylint: disable=unused-argument

    image_bytes = await file.read()
    try:
        config = MosaicConfig(
            title=str(title),
            num_segments=num_segments,
            mosaic_bg_brightness=mosaic_bg_brightness,
            mosaic_blend_value=mosaic_blend_value,
            segment_blend_value=segment_blend_value,
            segment_blur_low=segment_blur_low,
            segment_blur_medium=segment_blur_medium,
            segment_blur_high=segment_blur_high,
        )
        mosaic_id = mgmt_service.create_mosaic(image_bytes, config)
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
        mgmt_service.reset_mosaic(m_id)
        return JSONResponse(content={"msg": "Mosaic reset!"}, headers={"mosaic_id": m_id})
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
        mgmt_service.delete_mosaic(m_id)
        return JSONResponse(content={"msg": "Mosaic deleted!"}, headers={"mosaic_id": m_id})
    except HTTPException:
        raise
    except BaseException as exc:
        logging.error("", exc_info=True)
        raise HTTPException(status_code=500, detail="An unknown server error occurred") from exc


@router.post(
    "/mosaic/{mosaic_id}/states",
    name="update_mosaic_state",
    summary="Update the state of a mosaic. This might lead to invalid application states and unpredicted mosaic"
    " lifecycle behaviour.",
)
async def update_mosaic_state(
    mosaic_id: str,
    active: bool = Form(False, description="Mark the mosaic as active. Only one mosaic" " should be active."),
    filled: bool = Form(
        False, description="Mark the mosaic as filled. Indictates that no more segments shall be" " filled in a mosaic."
    ),
    original: bool = Form(False, description="Mark a mosaic as Original (created by the user)."),
    key: APIKey = Depends(auth_service.admin_auth),
) -> JSONResponse:
    # pylint: disable=unused-argument
    try:
        m_id = validate_request_uuid(mosaic_id, "mosaic")
        mgmt_service.update_mosaic_state(m_id, active, filled, original)
        return JSONResponse(content={"msg": "Mosaic state updated!"}, headers={"mosaic_id": m_id})
    except HTTPException:
        raise
    except BaseException as exc:
        logging.error("", exc_info=True)
        raise HTTPException(status_code=500, detail="An unknown server error occurred") from exc


@router.get(
    "/mosaic/{mosaic_id}/segment/list",
    name="list_segments",
    summary="List the ids of all segments in the mosaic.",
)
async def list_segments(mosaic_id: str, key: APIKey = Depends(auth_service.admin_auth)) -> JSONResponse:
    # pylint: disable=unused-argument
    try:
        m_id = validate_request_uuid(mosaic_id, "mosaic")
        segment_list = mgmt_service.get_segment_list(mosaic_id)
        return JSONResponse(content={"segment_list": segment_list}, headers={"mosaic_id": m_id})
    except HTTPException:
        raise
    except BaseException as exc:
        logging.error("", exc_info=True)
        raise HTTPException(status_code=500, detail="An unknown server error occurred") from exc


@router.post(
    "/mosaic/{mosaic_id}/segment/{segment_id}/reset",
    name="reset_segment",
    summary="Reset a single segment of a mosaic to its state after creation.",
)
async def reset_segment(
    mosaic_id: str, segment_id: str, key: APIKey = Depends(auth_service.admin_auth)
) -> JSONResponse:
    # pylint: disable=unused-argument
    try:
        m_id = validate_request_uuid(mosaic_id, "mosaic")
        s_id = validate_request_uuid(segment_id, "segment")
        mgmt_service.reset_segment(m_id, s_id)
        return JSONResponse(content={"msg": "Segment reset!"}, headers={"mosaic_id": m_id, "segment_id": s_id})
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
    quick_fill: bool = Form(
        True, description="If enabled each uploaded image will be filled into 5 different segemnts"
    ),
    file: UploadFile = File(..., description="A portrait image that shall be added to the mosaic."),
    api_key: APIKey = Depends(auth_service.admin_auth),
) -> JSONResponse:
    # pylint: disable=unused-argument
    try:
        input_image_bytes = await file.read()
        m_id = validate_request_uuid(mosaic_id, "mosaic")
        filling_service.fill_random_segments(m_id, input_image_bytes, quick_fill)
        return JSONResponse(content={"msg": "Segment filled!"}, headers={"mosaic_id": m_id})
    except HTTPException:
        raise
    except BaseException as exc:
        logging.error("", exc_info=True)
        raise HTTPException(status_code=500, detail="An unknown server error occurred") from exc
