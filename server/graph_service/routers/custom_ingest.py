"""
Dynamic ingestion router for meeting transcripts.
Uses Graphiti's default LLM-based entity/relationship extraction.
Entity and relationship types are dynamically determined by the LLM.
"""
import asyncio
from typing import Dict, Optional

from fastapi import APIRouter, status
from graphiti_core.nodes import EpisodeType
from pydantic import BaseModel, Field

from graph_service.dto import Result
from graph_service.zep_graphiti import ZepGraphitiDep


# ============================================================================
# Request/Response Models
# ============================================================================

class AddMeetingEpisodeRequest(BaseModel):
    """Request model for adding a meeting episode"""
    name: str = Field(description="Episode name")
    episode_body: str = Field(description="Meeting content/transcript with extracted entities")
    group_id: str = Field(description="Group ID (e.g., orgId-meetingId)")
    source_description: str = Field(default="meeting_transcript", description="Source description")
    reference_time: Optional[str] = Field(default=None, description="ISO datetime when episode occurred")
    metadata: Optional[Dict] = Field(default=None, description="Additional metadata")


# ============================================================================
# Router Setup
# ============================================================================

router = APIRouter(prefix="/meeting", tags=["meeting-ingest"])


# ============================================================================
# Endpoint: Add Meeting Episode (Dynamic Entity Extraction)
# ============================================================================

@router.post('/episode', status_code=status.HTTP_202_ACCEPTED)
async def add_meeting_episode(
    request: AddMeetingEpisodeRequest,
    graphiti: ZepGraphitiDep,
):
    """
    Add a meeting episode for knowledge graph processing.

    This endpoint uses Graphiti's default LLM-based extraction which:
    - Dynamically determines entity types from the content
    - Dynamically determines relationship types from the content
    - No predefined schema - AI decides everything

    The episode_body should contain the meeting transcript along with
    any pre-extracted entities/relationships formatted as structured text.
    Graphiti's LLM will process and create the knowledge graph.

    Returns:
        202 Accepted with message indicating episode was queued for processing
    """

    async def add_episode_task():
        # Use Graphiti's default extraction - no custom entity/edge types
        # This lets Graphiti's LLM dynamically determine types
        await graphiti.add_episode(
            name=request.name,
            episode_body=request.episode_body,
            group_id=request.group_id,
            source=EpisodeType.text,
            source_description=request.source_description,
            reference_time=request.reference_time,
            # NO entity_types, edge_types, or edge_type_map
            # Graphiti will use its default LLM-based extraction
        )

    # Queue the task for async processing
    asyncio.create_task(add_episode_task())

    return Result(
        message=f'Meeting episode "{request.name}" added to processing queue (dynamic extraction)',
        success=True
    )


# ============================================================================
# Endpoint: Health Check
# ============================================================================

@router.get('/health')
async def meeting_health():
    """Health check endpoint for meeting ingestion service"""
    return {
        'status': 'healthy',
        'mode': 'dynamic',
        'description': 'Entity and relationship types are dynamically determined by LLM'
    }
