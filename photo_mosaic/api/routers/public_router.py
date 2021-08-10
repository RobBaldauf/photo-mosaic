import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from photo_mosaic.services.mosaic_filling import filling_service
from photo_mosaic.services.mosaic_management import mgmt_service
from photo_mosaic.utils.request_validation import validate_request_uuid

router = APIRouter()


@router.get("/mosaic/list", name="list_all_mosaics", summary="List the ids of all available mosaics.")
async def list_all_mosaics() -> JSONResponse:
    try:
        mosaic_list = mgmt_service.get_mosaic_list(filter_by="ALL")
        return JSONResponse(content={"mosaic_list": mosaic_list})
    except HTTPException:
        raise
    except BaseException as exc:
        logging.error("", exc_info=True)
        raise HTTPException(status_code=500, detail="An unknown server error occurred") from exc


@router.get("/mosaic/list/active", name="list_active_mosaics", summary="List the id of the active mosaic.")
async def list_active_mosaics() -> JSONResponse:
    try:
        mosaic_list = mgmt_service.get_mosaic_list(filter_by="ACTIVE")
        return JSONResponse(content={"mosaic_list": mosaic_list})
    except HTTPException:
        raise
    except BaseException as exc:
        logging.error("", exc_info=True)
        raise HTTPException(status_code=500, detail="An unknown server error occurred") from exc


@router.get(
    "/mosaic/list/original",
    name="list_original_mosaics",
    summary="List the ids of all original mosaics (Mosaics that " "were manually created by the user).",
)
async def list_original_mosaics() -> JSONResponse:
    try:
        mosaic_list = mgmt_service.get_mosaic_list(filter_by="ORIGINAL")
        return JSONResponse(content={"mosaic_list": mosaic_list})
    except HTTPException:
        raise
    except BaseException as exc:
        logging.error("", exc_info=True)
        raise HTTPException(status_code=500, detail="An unknown server error occurred") from exc


@router.get(
    "/mosaic/list/filled",
    name="list_filled_mosaics",
    summary="List the ids of all mosaics for which every " "segment has already been filled with an image.",
)
async def list_filled_mosaics() -> JSONResponse:
    try:
        mosaic_list = mgmt_service.get_mosaic_list(filter_by="FILLED")
        return JSONResponse(content={"mosaic_list": mosaic_list})
    except HTTPException:
        raise
    except BaseException as exc:
        logging.error("", exc_info=True)
        raise HTTPException(status_code=500, detail="An unknown server error occurred") from exc


@router.get(
    "/mosaic/{mosaic_id}", name="get_mosaic_current", summary="Get the current state of a mosaic as a JPEG " "image."
)
async def get_mosaic_current(mosaic_id: str) -> Response:
    try:
        m_id = validate_request_uuid(mosaic_id, "mosaic")
        image_bytes = mgmt_service.get_mosaic_current_jpeg(m_id)
        return Response(content=image_bytes, media_type="image/jpeg", headers={"mosaic_id": m_id})
    except HTTPException:
        raise
    except BaseException as exc:
        logging.error("", exc_info=True)
        raise HTTPException(status_code=500, detail="An unknown server error occurred") from exc


@router.get(
    "/mosaic/{mosaic_id}/thumbnail",
    name="get_mosaic_current_thumbnail",
    summary="Get the current state of a mosaic as a JPEG thumbnail image (e.g. for a gallery).",
)
async def get_mosaic_current_thumbnail(mosaic_id: str) -> Response:
    try:
        m_id = validate_request_uuid(mosaic_id, "mosaic")
        image_bytes = mgmt_service.get_mosaic_current_jpeg_thumbnail(m_id)
        return Response(content=image_bytes, media_type="image/jpeg", headers={"mosaic_id": m_id})
    except HTTPException:
        raise
    except BaseException as exc:
        logging.error("", exc_info=True)
        raise HTTPException(status_code=500, detail="An unknown server error occurred") from exc


@router.get("/mosaic/{mosaic_id}/metadata", name="get_mosaic_metadata", summary="Get the metadata of a mosaic as JSON.")
async def get_mosaic_metadata(mosaic_id: str) -> JSONResponse:
    try:
        m_id = validate_request_uuid(mosaic_id, "mosaic")
        metadata = mgmt_service.get_mosaic_metadata(m_id)
        return JSONResponse(content=metadata, headers={"mosaic_id": m_id})
    except HTTPException:
        raise
    except BaseException as exc:
        logging.error("", exc_info=True)
        raise HTTPException(status_code=500, detail="An unknown server error occurred") from exc


@router.get(
    "/mosaic/{mosaic_id}/original",
    name="get_mosaic_original",
    summary="Get the original image that was used to create a mosaic.",
)
async def get_mosaic_original(mosaic_id: str) -> Response:
    try:
        m_id = validate_request_uuid(mosaic_id, "mosaic")
        image_bytes = mgmt_service.get_mosaic_original_jpeg(m_id)
        return Response(content=image_bytes, media_type="image/jpeg", headers={"mosaic_id": m_id})
    except HTTPException:
        raise
    except BaseException as exc:
        logging.error("", exc_info=True)
        raise HTTPException(status_code=500, detail="An unknown server error occurred") from exc


@router.get(
    "/mosaic/{mosaic_id}/gif",
    name="get_mosaic_gif",
    summary="Get the filling process for a mosaic animated as a gif.",
)
async def get_mosaic_gif(mosaic_id: str) -> Response:
    try:
        m_id = validate_request_uuid(mosaic_id, "mosaic")
        image_bytes = mgmt_service.get_mosaic_filling_gif(m_id)
        return Response(content=image_bytes, media_type="image/gif", headers={"mosaic_id": m_id})
    except HTTPException:
        raise
    except BaseException as exc:
        logging.error("", exc_info=True)
        raise HTTPException(status_code=500, detail="An unknown server error occurred") from exc


@router.post(
    "/mosaic/{mosaic_id}/segment/sample",
    name="get_segment_samples",
    summary="For an uploaded image, select a matching mosaic segment (based on brightness and position), apply the "
    "segment as a 'filter' to the uploaded image and return the resulting image as well as the id of the used "
    "segment. This will not change the mosaic.",
)
async def post_segment_sample(
    mosaic_id: str,
    file: UploadFile = File(
        ...,
        description="A portrait image that shall be uploaded, merged with a segment and "
        "returned as a 'stylised' version.",
    ),
    sample_index: int = Form(
        0,
        ge=0,
        le=65536,
        description="The index of the random sample. Different values return different filter samples",
    ),
) -> Response:
    input_image_bytes = await file.read()
    try:
        m_id = validate_request_uuid(mosaic_id, "mosaic")
        output_image_bytes, segment_id = filling_service.get_segment_sample(m_id, input_image_bytes, sample_index)
        return Response(
            content=output_image_bytes,
            media_type="image/jpeg",
            headers={"mosaic_id": m_id, "segment_id": segment_id},
        )

    except HTTPException:
        raise
    except BaseException as exc:
        logging.error("", exc_info=True)
        raise HTTPException(status_code=500, detail="An unknown server error occurred") from exc


@router.post(
    "/mosaic/{mosaic_id}/segment/{segment_id}",
    name="post_mosaic_segment",
    summary="For an uploaded image and segment_id, apply the segment as a 'filter' to the uploaded image and add the "
    "result to the mosaic. This will update the mosaic and all related data structures.",
)
async def post_mosaic_segment(
    mosaic_id: str,
    segment_id: str,
    file: UploadFile = File(
        ..., description="A portrait image that shall be merged with a segment and added " "to the mosaic."
    ),
) -> JSONResponse:
    try:
        input_image_bytes = await file.read()
        m_id = validate_request_uuid(mosaic_id, "mosaic")
        seg_id = validate_request_uuid(segment_id, "segment")
        filling_service.fill_segment(m_id, input_image_bytes, seg_id)
        return JSONResponse(content={"msg": "Segment filled!"}, headers={"mosaic_id": m_id})
    except HTTPException:
        raise
    except BaseException as exc:
        logging.error("", exc_info=True)
        raise HTTPException(status_code=500, detail="An unknown server error occurred") from exc
