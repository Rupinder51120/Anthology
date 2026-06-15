from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from api.core.config import get_settings

settings = get_settings()


class Base(DeclarativeBase):
    pass


def get_database_url() -> str:
    url = settings.database_url
    # Render provides postgresql:// — convert to asyncpg
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


engine = create_async_engine(
    get_database_url(),
    echo=settings.debug,
    pool_pre_ping=True,
    pool_size=5,       # FIX: was 1 — too conservative for concurrent requests
    max_overflow=10,   # FIX: was 0
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
