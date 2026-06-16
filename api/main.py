from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.core.config import get_settings
from api.core.database import create_tables
from api.routers import health, query, papers, search, recommend, tts, flowchart, benchmark, feedback, stats, discovery, sessions, suggest, collections

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        await create_tables()
        # Preload models to avoid cold start latency
        from src.retrieval.embedder import get_model
        get_model()
        print(f"Anthology API v{settings.app_version} started")
    except Exception as e:
        print(f"Startup warning: {e}")
    yield
    # Shutdown
    print("Anthology API shutting down")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI Research Intelligence System — RAG over 119+ research papers",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router)
app.include_router(query.router)
app.include_router(papers.router)
app.include_router(search.router)
app.include_router(recommend.router)
app.include_router(tts.router)
app.include_router(flowchart.router)
app.include_router(benchmark.router)
app.include_router(feedback.router)
app.include_router(stats.router)
app.include_router(discovery.router)
app.include_router(sessions.router)
app.include_router(suggest.router)
app.include_router(collections.router)


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    }
