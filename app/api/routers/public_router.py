import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.openapi.models import APIKey
from fastapi.responses import JSONResponse, Response

from app.models.app_config import get_config
from app.services.auth import AuthService
from app.services.mail import MailService
from app.services.mosaic_filling import MosaicFillingService
from app.services.mosaic_management import MosaicManagementService
from app.services.nsfw import NSFWService
from app.services.sqlite_persistence import SQLiteAbstractPersistenceService
from app.utils.data import validate_request_uuid

router = APIRouter()
auth_service = AuthService()
persistence_service = SQLiteAbstractPersistenceService(get_config().sql_lite_path)
if get_config().enable_nsfw_content_filter:
    filling_service = MosaicFillingService(persistence_service, NSFWService(get_config().nsfw_model_path))
else:
    filling_service = MosaicFillingService(persistence_service)
mgmt_service = MosaicManagementService(persistence_service)
mail_service = MailService(persistence_service)


@router.get("/mosaic/list", name="list_all_mosaics", summary="List the ids of all available mosaics.")
async def list_all_mosaics(key: APIKey = Depends(auth_service.public_auth)) -> JSONResponse:
    # pylint: disable=unused-argument
    try:
        mosaic_list = await mgmt_service.get_mosaic_list(filter_by="ALL")
        return JSONResponse(content={"mosaic_list": mosaic_list})
    except HTTPException:
        raise
    except BaseException as exc:
        logging.error("", exc_info=True)
        raise HTTPException(status_code=500, detail="An unknown server error occurred") from exc


@router.get("/mosaic/list/active", name="list_active_mosaics", summary="List the id of the active mosaic.")
async def list_active_mosaics(key: APIKey = Depends(auth_service.public_auth)) -> JSONResponse:
    # pylint: disable=unused-argument
    try:
        mosaic_list = await mgmt_service.get_mosaic_list(filter_by="ACTIVE")
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
async def list_original_mosaics(key: APIKey = Depends(auth_service.public_auth)) -> JSONResponse:
    # pylint: disable=unused-argument
    try:
        mosaic_list = await mgmt_service.get_mosaic_list(filter_by="ORIGINAL")
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
async def list_filled_mosaics(key: APIKey = Depends(auth_service.public_auth)) -> JSONResponse:
    # pylint: disable=unused-argument
    try:
        mosaic_list = await mgmt_service.get_mosaic_list(filter_by="FILLED")
        return JSONResponse(content={"mosaic_list": mosaic_list})
    except HTTPException:
        raise
    except BaseException as exc:
        logging.error("", exc_info=True)
        raise HTTPException(status_code=500, detail="An unknown server error occurred") from exc


@router.get(
    "/mosaic/{mosaic_id}", name="get_mosaic_current", summary="Get the current state of a mosaic as a JPEG " "image."
)
async def get_mosaic_current(mosaic_id: str, key: APIKey = Depends(auth_service.public_auth)) -> Response:
    # pylint: disable=unused-argument
    try:
        m_id = validate_request_uuid(mosaic_id, "mosaic")
        image_bytes = await mgmt_service.get_mosaic_current_jpeg(m_id)
        return Response(content=image_bytes, media_type="image/jpeg", headers={"mosaic_id": m_id})
    except HTTPException:
        raise
    except BaseException as exc:
        logging.error("", exc_info=True)
        raise HTTPException(status_code=500, detail="An unknown server error occurred") from exc


@router.get("/mosaic/{mosaic_id}/metadata", name="get_mosaic_metadata", summary="Get the metadata of a mosaic as JSON.")
async def get_mosaic_metadata(mosaic_id: str, key: APIKey = Depends(auth_service.public_auth)) -> JSONResponse:
    # pylint: disable=unused-argument
    try:
        m_id = validate_request_uuid(mosaic_id, "mosaic")
        metadata = await mgmt_service.get_mosaic_metadata(m_id)
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
async def get_mosaic_original(mosaic_id: str, key: APIKey = Depends(auth_service.public_auth)) -> Response:
    # pylint: disable=unused-argument
    try:
        m_id = validate_request_uuid(mosaic_id, "mosaic")
        image_bytes = await mgmt_service.get_mosaic_original_jpeg(m_id)
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
async def get_mosaic_gif(mosaic_id: str, key: APIKey = Depends(auth_service.public_auth)) -> Response:
    # pylint: disable=unused-argument
    try:
        m_id = validate_request_uuid(mosaic_id, "mosaic")
        image_bytes = await mgmt_service.get_mosaic_filling_gif(m_id)
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
    mosaic_id: str, file: UploadFile = File(...), key: APIKey = Depends(auth_service.public_auth)
) -> Response:
    # pylint: disable=unused-argument
    input_image_bytes = await file.read()
    try:
        m_id = validate_request_uuid(mosaic_id, "mosaic")
        output_image_bytes, segment_id = await filling_service.get_segment_sample(m_id, input_image_bytes)
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
    file: UploadFile = File(...),
    key: APIKey = Depends(auth_service.public_auth),
) -> JSONResponse:
    # pylint: disable=unused-argument
    try:
        input_image_bytes = await file.read()
        m_id = validate_request_uuid(mosaic_id, "mosaic")
        seg_id = validate_request_uuid(segment_id, "segment")
        await filling_service.fill_segment(m_id, input_image_bytes, seg_id)
        return JSONResponse(content={"msg": "Segment filled!"}, headers={"mosaic_id": m_id})
    except HTTPException:
        raise
    except BaseException as exc:
        logging.error("", exc_info=True)
        raise HTTPException(status_code=500, detail="An unknown server error occurred") from exc


# @router.post(
#     "/mosaic/{mosaic_id}/segment/email",
#     name="post_segment_email",
#     summary="Takes a segment image and sends it via email to the provided recipient.",
# )
# async def post_segment_email(
#         segment_id: str, mail_adr: str, file: UploadFile = File(...), key: APIKey = Depends(auth_service.public_auth)
# ) -> Response:
#     # pylint: disable=unused-argument
#     try:
#         s_id = validate_request_uuid(segment_id, "segment")
#         await mail_service.send_filtered_image_email(file, mail_adr)
#         return JSONResponse(content={"msg": "Email sent!"}, headers={"segment_id": s_id})
#
#     except HTTPException:
#         raise
#     except BaseException as exc:
#         logging.error("", exc_info=True)
#         raise HTTPException(status_code=500, detail="An unknown server error occurred") from exc
