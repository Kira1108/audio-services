import logging
logging.basicConfig(level = logging.INFO)

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from app import routers
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return JSONResponse(content={"status": "ok"})

app.include_router(routers.asr.router)
