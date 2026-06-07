from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from api.core.database import get_db
from api.core.config import get_settings, Settings


def get_settings_dep() -> Settings:
    return get_settings()
