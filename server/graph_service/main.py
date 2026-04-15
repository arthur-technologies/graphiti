import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from graph_service.config import get_settings
from graph_service.routers import ingest, retrieve, custom_ingest
from graph_service.zep_graphiti import initialize_graphiti


def configure_logging() -> None:
    log_level_name = os.getenv('GRAPHITI_LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    logging.getLogger('graphiti_core').setLevel(log_level)
    logging.getLogger('graph_service').setLevel(log_level)


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    settings = get_settings()
    await initialize_graphiti(settings)
    yield
    # Shutdown
    # No need to close Graphiti here, as it's handled per-request


app = FastAPI(lifespan=lifespan)


app.include_router(retrieve.router)
app.include_router(ingest.router)
app.include_router(custom_ingest.router)


@app.get('/healthcheck')
async def healthcheck():
    return JSONResponse(content={'status': 'healthy'}, status_code=200)
