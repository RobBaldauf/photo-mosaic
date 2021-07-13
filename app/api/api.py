from fastapi import APIRouter

from app.api.routers import admin_router, public_router

api = APIRouter()
api.include_router(public_router.router, tags=["public"])
api.include_router(admin_router.router, tags=["admin"])
