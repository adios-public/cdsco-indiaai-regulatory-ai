from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import anonymisation, summarisation, completeness, classification, inspection

app = FastAPI(
    title="AdiOS Regulatory AI",
    description="CDSCO-IndiaAI Hackathon — AI-driven regulatory workflow automation",
    version="1.0.0",
    contact={
        "name": "AdiOS Platform Pvt Ltd",
        "email": "malay@adiosplat.io",
        "url": "https://www.adiosplat.io",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(anonymisation.router, prefix="/api/v1")
app.include_router(summarisation.router, prefix="/api/v1")
app.include_router(completeness.router, prefix="/api/v1")
app.include_router(classification.router, prefix="/api/v1")
app.include_router(inspection.router, prefix="/api/v1")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "adios-regulatory-ai"}
