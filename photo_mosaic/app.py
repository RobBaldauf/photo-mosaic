""" Main Server Script"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from photo_mosaic.api.api import api
from photo_mosaic.models.app_config import get_config
from photo_mosaic.services.persistence import db
from photo_mosaic.utils.version import version


# Check and display important api settings
def check_config() -> str:
    documentation_url = None
    if get_config().enable_documentation:
        documentation_url = "/documentation"
        print("Documentation endpoint: ENABLED")
    else:
        print("Documentation endpoint: DISABLED")

    if get_config().enable_auth:
        if not get_config().jwt_secret:
            raise ValueError("Please set JWT_SECRET via '.env' file or environment variable!")
        print("Authentication for admin endpoints: ENABLED")
    else:
        print("Authentication for admin endpoints: DISABLED")

    if get_config().enable_nsfw_content_filter:
        if not get_config().nsfw_model_path:
            raise ValueError("Please set NSFW_MODEL_PATH via '.env' file or environment variable!")
        print("NSFW content filtering: ENABLED")
    else:
        print("NSFW content filtering: DISABLED")

    if not get_config().sqlite_path:
        raise ValueError("Please set SQL_LITE_PATH via '.env' file or environment variable!")
    if get_config().uploaded_image_path:
        print(f"Uploaded images will be stored to {get_config().uploaded_image_path}")
    else:
        print("UPLOADED_IMAGE_PATH is not set. Uploaded images will not be stored.")
    return documentation_url


docs_url = check_config()

# setup CORS middleware
middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=get_config().cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["mosaic_id", "segment_id"],
    )
]

# setup api server
app = FastAPI(
    title="Photo Mosaic API",
    version=version(),
    middleware=middleware,
    docs_url=docs_url,
    redoc_url=None,
)
app.include_router(router=api)
Instrumentator().instrument(app).expose(app, include_in_schema=False)


@app.on_event("startup")
async def database_connect():
    print(f"Running photo-mosaic service (v{version()})...")
    db.connect()


@app.on_event("shutdown")
async def database_disconnect():
    print("Stopping photo-mosaic service...")
    db.disconnect()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8111, workers=1)
