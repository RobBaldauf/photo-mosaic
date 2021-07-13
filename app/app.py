""" Main Server Script"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware

from app.api.api import api
from app.models.app_config import get_config
from app.utils.utils import version

docs_url = None
if get_config().enable_documentation:
    docs_url = "/documentation"
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

if not get_config().sql_lite_path:
    raise ValueError("Please set SQL_LITE_PATH via '.env' file or environment variable!")

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
app = FastAPI(title="Pixel Mosaic API", version=version(), middleware=middleware, docs_url=docs_url, redoc_url=None)
app.include_router(router=api)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8111)
