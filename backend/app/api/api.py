from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.endpoints import router as api_v1_router
from app.core.config import settings

app = FastAPI(
    title="Aviation Weather API Hub",
    description="A master API to manage and reference aviation weather APIs",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(api_v1_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Aviation Weather API Hub",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "version": "0.1.0"
    }
