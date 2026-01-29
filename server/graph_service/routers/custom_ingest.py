"""
Dynamic ingestion router for meeting transcripts.
Uses Graphiti's default LLM-based entity/relationship extraction.
Entity and relationship types are dynamically determined by the LLM.

Supports both single episode and bulk episode ingestion:
- /meeting/episode: Single episode (fire-and-forget, may cause duplicates if called rapidly)
- /meeting/episodes/bulk: Bulk episodes (recommended - handles deduplication properly)
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, status
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.bulk_utils import RawEpisode
from pydantic import BaseModel, Field

from graph_service.dto import Result
from graph_service.zep_graphiti import ZepGraphitiDep

logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================

class AddMeetingEpisodeRequest(BaseModel):
    """Request model for adding a meeting episode"""
    name: str = Field(description="Episode name")
    episode_body: str = Field(description="Meeting content/transcript with extracted entities")
    group_id: str = Field(description="Group ID (e.g., orgId)")
    source_description: str = Field(default="meeting_transcript", description="Source description")
    reference_time: Optional[str] = Field(default=None, description="ISO datetime when episode occurred")
    metadata: Optional[Dict] = Field(default=None, description="Additional metadata")
    custom_extraction_instructions: Optional[str] = Field(
        default=None,
        description="Custom instructions to guide LLM extraction (e.g., for user memory extraction)"
    )


class BulkEpisodeItem(BaseModel):
    """Single episode item within a bulk request"""
    name: str = Field(description="Episode name")
    episode_body: str = Field(description="Meeting content/transcript")
    source_description: str = Field(default="meeting_transcript", description="Source description")
    reference_time: Optional[str] = Field(default=None, description="ISO datetime when episode occurred")


class AddBulkEpisodesRequest(BaseModel):
    """Request model for adding multiple episodes in bulk"""
    episodes: List[BulkEpisodeItem] = Field(description="List of episodes to process")
    group_id: str = Field(description="Group ID (e.g., orgId) - shared by all episodes")


class BulkEpisodeResult(BaseModel):
    """Response model for bulk episode ingestion"""
    message: str
    success: bool
    data: Optional[Dict[str, Any]] = None


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
    # Parse reference_time from string to datetime
    if request.reference_time:
        try:
            ref_time = datetime.fromisoformat(request.reference_time.replace('Z', '+00:00'))
        except ValueError:
            ref_time = datetime.now(timezone.utc)
    else:
        ref_time = datetime.now(timezone.utc)

    async def add_episode_task():
        # Use Graphiti's default extraction - no custom entity/edge types
        # This lets Graphiti's LLM dynamically determine types
        await graphiti.add_episode(
            name=request.name,
            episode_body=request.episode_body,
            group_id=request.group_id,
            source=EpisodeType.text,
            source_description=request.source_description,
            reference_time=ref_time,
            custom_extraction_instructions=request.custom_extraction_instructions,
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
# Endpoint: Add Meeting Episode SYNCHRONOUSLY (for sequential processing)
# ============================================================================

@router.post('/episode/sync', status_code=status.HTTP_201_CREATED)
async def add_meeting_episode_sync(
    request: AddMeetingEpisodeRequest,
    graphiti: ZepGraphitiDep,
):
    """
    Add a meeting episode SYNCHRONOUSLY - waits for full processing before returning.

    Use this endpoint when sending paragraphs sequentially to ensure proper
    entity deduplication across paragraphs. Each call waits for Graphiti's
    LLM extraction and Neo4j persistence to complete.

    This is ideal for:
    - Sequential paragraph processing (one at a time)
    - Ensuring entity deduplication works correctly (previous episodes saved first)
    - Getting accurate node/edge counts per episode

    Returns:
        201 Created with episode UUID, nodes created, edges created, and processing time
    """
    # Parse reference_time from string to datetime
    if request.reference_time:
        try:
            ref_time = datetime.fromisoformat(request.reference_time.replace('Z', '+00:00'))
        except ValueError:
            ref_time = datetime.now(timezone.utc)
    else:
        ref_time = datetime.now(timezone.utc)

    start_time = datetime.now(timezone.utc)

    logger.info(f"Processing episode SYNCHRONOUSLY: {request.name} for group {request.group_id}")

    try:
        # Process SYNCHRONOUSLY - wait for completion
        result = await graphiti.add_episode(
            name=request.name,
            episode_body=request.episode_body,
            group_id=request.group_id,
            source=EpisodeType.text,
            source_description=request.source_description,
            reference_time=ref_time,
            custom_extraction_instructions=request.custom_extraction_instructions,
        )

        end_time = datetime.now(timezone.utc)
        duration_ms = (end_time - start_time).total_seconds() * 1000

        logger.info(
            f"Episode processed: {request.name} - "
            f"{len(result.nodes)} nodes, {len(result.edges)} edges in {duration_ms:.0f}ms"
        )

        return Result(
            message=f'Episode "{request.name}" processed successfully',
            success=True,
            data={
                'episode_uuid': result.episode.uuid,
                'nodes_created': len(result.nodes),
                'edges_created': len(result.edges),
                'processing_time_ms': round(duration_ms),
            }
        )

    except Exception as e:
        logger.error(f"Sync episode processing failed: {str(e)}", exc_info=True)
        return Result(
            message=f'Episode processing failed: {str(e)}',
            success=False,
            data={'error': str(e)}
        )


# ============================================================================
# Endpoint: Add Meeting Episodes in Bulk (RECOMMENDED)
# ============================================================================

@router.post('/episodes/bulk', status_code=status.HTTP_201_CREATED)
async def add_meeting_episodes_bulk(
    request: AddBulkEpisodesRequest,
    graphiti: ZepGraphitiDep,
) -> BulkEpisodeResult:
    """
    Add multiple meeting episodes for knowledge graph processing in a single batch.

    This is the RECOMMENDED endpoint for ingesting meeting content because:
    - Handles entity deduplication WITHIN the batch (in-memory)
    - Prevents duplicate entities that occur with rapid sequential /episode calls
    - More efficient - single transaction to Neo4j
    - Parallel LLM processing with proper deduplication

    How it works:
    1. All episodes are extracted in parallel (entity + edge extraction)
    2. In-memory deduplication merges identical entities across episodes
    3. Single bulk save to Neo4j with deduplicated entities

    Args:
        request: Contains list of episodes and shared group_id

    Returns:
        201 Created with statistics about processed episodes, nodes, and edges
    """
    start_time = datetime.now(timezone.utc)

    if not request.episodes:
        return BulkEpisodeResult(
            message="No episodes provided",
            success=False,
            data={"episodes": 0, "nodes": 0, "edges": 0}
        )

    logger.info(
        f"Starting bulk ingestion of {len(request.episodes)} episodes for group {request.group_id}"
    )

    try:
        # Convert to RawEpisode objects for Graphiti bulk API
        bulk_episodes: List[RawEpisode] = []
        for ep in request.episodes:
            # Parse reference_time
            if ep.reference_time:
                try:
                    ref_time = datetime.fromisoformat(ep.reference_time.replace('Z', '+00:00'))
                except ValueError:
                    ref_time = datetime.now(timezone.utc)
            else:
                ref_time = datetime.now(timezone.utc)

            bulk_episodes.append(
                RawEpisode(
                    name=ep.name,
                    content=ep.episode_body,
                    source=EpisodeType.text,
                    source_description=ep.source_description,
                    reference_time=ref_time,
                )
            )

        # Call Graphiti's bulk API - this handles deduplication internally
        result = await graphiti.add_episode_bulk(
            bulk_episodes=bulk_episodes,
            group_id=request.group_id,
        )

        end_time = datetime.now(timezone.utc)
        duration_ms = (end_time - start_time).total_seconds() * 1000

        logger.info(
            f"Bulk ingestion complete: {len(result.episodes)} episodes, "
            f"{len(result.nodes)} nodes, {len(result.edges)} edges in {duration_ms:.0f}ms"
        )

        return BulkEpisodeResult(
            message=f"Bulk ingestion complete: {len(result.episodes)} episodes processed",
            success=True,
            data={
                "episodes_processed": len(result.episodes),
                "nodes_created": len(result.nodes),
                "edges_created": len(result.edges),
                "processing_time_ms": round(duration_ms),
                "group_id": request.group_id,
            }
        )

    except Exception as e:
        logger.error(f"Bulk ingestion failed: {str(e)}", exc_info=True)
        return BulkEpisodeResult(
            message=f"Bulk ingestion failed: {str(e)}",
            success=False,
            data={"error": str(e)}
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
        'description': 'Entity and relationship types are dynamically determined by LLM',
        'endpoints': {
            '/meeting/episode': 'Single episode (fire-and-forget, async)',
            '/meeting/episode/sync': 'Single episode (synchronous - waits for completion)',
            '/meeting/episodes/bulk': 'Bulk episodes (in-memory deduplication)',
        }
    }
