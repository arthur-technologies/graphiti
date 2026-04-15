from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, status
from graphiti_core.helpers import parse_db_date
from graphiti_core.nodes import EpisodicNode
from graphiti_core.search.search_config import (
    EdgeReranker,
    EdgeSearchConfig,
    EdgeSearchMethod,
    NodeReranker,
    NodeSearchConfig,
    NodeSearchMethod,
    CommunityReranker,
    CommunitySearchConfig,
    CommunitySearchMethod,
    SearchConfig,
)

from graph_service.dto import (
    ComprehensiveSearchQuery,
    ComprehensiveSearchResults,
    CommunityResult,
    EpisodeSourceResult,
    EntityResult,
    GetMemoryRequest,
    GetMemoryResponse,
    Message,
    SearchQuery,
    SearchResults,
    SourceResults,
)
from graph_service.zep_graphiti import ZepGraphitiDep, get_fact_result_from_edge

router = APIRouter()

# Tuned search config - balanced between precision and recall
# No BFS to avoid graph traversal noise, lowered similarity threshold for better recall
MEETING_SEARCH_CONFIG = SearchConfig(
    edge_config=EdgeSearchConfig(
        search_methods=[EdgeSearchMethod.bm25, EdgeSearchMethod.cosine_similarity],
        reranker=EdgeReranker.rrf,  # RRF is more permissive than MMR
        sim_min_score=0.3,  # Lower threshold for better recall
    ),
    node_config=NodeSearchConfig(
        search_methods=[NodeSearchMethod.bm25, NodeSearchMethod.cosine_similarity],
        reranker=NodeReranker.rrf,
        sim_min_score=0.3,
    ),
    community_config=CommunitySearchConfig(
        search_methods=[CommunitySearchMethod.bm25, CommunitySearchMethod.cosine_similarity],
        reranker=CommunityReranker.rrf,
        sim_min_score=0.3,
    ),
    limit=10,
    reranker_min_score=0.0,  # Don't filter by reranker score
)


def get_entity_result(node) -> EntityResult:
    """Convert EntityNode to EntityResult"""
    return EntityResult(
        uuid=node.uuid,
        name=node.name,
        entity_type=getattr(node, 'labels', [None])[0] if hasattr(node, 'labels') and node.labels else None,
        summary=getattr(node, 'summary', None),
        created_at=getattr(node, 'created_at', None),
        attributes=getattr(node, 'attributes', {}) or {},
    )


def get_community_result(community) -> CommunityResult:
    """Convert CommunityNode to CommunityResult"""
    return CommunityResult(
        uuid=community.uuid,
        name=community.name,
        summary=getattr(community, 'summary', None),
        created_at=getattr(community, 'created_at', None),
    )


def get_episode_source_result(
    episode: EpisodicNode,
    matched_entity_count: int | None = None,
    matched_entity_names: list[str] | None = None,
) -> EpisodeSourceResult:
    source = getattr(episode, 'source', None)
    return EpisodeSourceResult(
        uuid=episode.uuid,
        name=episode.name,
        source=source.value if hasattr(source, 'value') else source,
        source_description=getattr(episode, 'source_description', None),
        content=getattr(episode, 'content', None),
        valid_at=getattr(episode, 'valid_at', None),
        created_at=getattr(episode, 'created_at', None),
        matched_entity_count=matched_entity_count,
        matched_entity_names=matched_entity_names or [],
    )


def sort_episode_sources(sources: list[EpisodeSourceResult]) -> list[EpisodeSourceResult]:
    return sorted(
        sources,
        key=lambda source: (
            source.valid_at or source.created_at or datetime.min.replace(tzinfo=timezone.utc)
        ),
        reverse=True,
    )


@router.post('/search', status_code=status.HTTP_200_OK)
async def search(query: SearchQuery, graphiti: ZepGraphitiDep):
    relevant_edges = await graphiti.search(
        group_ids=query.group_ids,
        query=query.query,
        num_results=query.max_facts,
    )
    facts = [get_fact_result_from_edge(edge) for edge in relevant_edges]
    return SearchResults(
        facts=facts,
    )


@router.post('/search-all', status_code=status.HTTP_200_OK)
async def search_all(query: ComprehensiveSearchQuery, graphiti: ZepGraphitiDep):
    """
    Comprehensive search across all graph layers: facts, entities, and communities.

    This endpoint searches:
    - Facts/Edges: Relationships between entities
    - Entities/Nodes: People, projects, technologies with summaries
    - Communities: Cluster summaries for high-level context

    Uses tuned search config to reduce noise while maintaining good recall.
    """
    # Create config with requested limit
    config = SearchConfig(
        edge_config=MEETING_SEARCH_CONFIG.edge_config if query.include_facts else None,
        node_config=MEETING_SEARCH_CONFIG.node_config if query.include_entities else None,
        community_config=MEETING_SEARCH_CONFIG.community_config if query.include_communities else None,
        limit=query.max_results,
        reranker_min_score=MEETING_SEARCH_CONFIG.reranker_min_score,
    )

    # Use search_ for comprehensive results
    results = await graphiti.search_(
        query=query.query,
        config=config,
        group_ids=query.group_ids,
    )

    # Convert to response format
    facts = [get_fact_result_from_edge(edge) for edge in results.edges] if query.include_facts else []
    entities = [get_entity_result(node) for node in results.nodes] if query.include_entities else []
    communities = [get_community_result(comm) for comm in results.communities] if query.include_communities else []

    return ComprehensiveSearchResults(
        facts=facts,
        entities=entities,
        communities=communities,
    )


@router.get('/sources/{item_type}/{uuid}', status_code=status.HTTP_200_OK)
async def get_sources(
    item_type: str,
    uuid: str,
    graphiti: ZepGraphitiDep,
    group_id: str = Query(..., description='The group/org id to scope source lookup'),
    limit: int = Query(12, ge=1, le=50),
):
    normalized_type = item_type.lower()
    if normalized_type not in {'fact', 'entity', 'community', 'topic'}:
        raise HTTPException(status_code=400, detail='Unsupported item type')

    if normalized_type == 'fact':
        edge = await graphiti.get_entity_edge(uuid)
        episode_uuids = list(dict.fromkeys(edge.episodes or []))
        if not episode_uuids:
            return SourceResults(item_type='fact', item_uuid=uuid, sources=[])

        episodes = await EpisodicNode.get_by_uuids(graphiti.driver, episode_uuids)
        sources = sort_episode_sources([
            get_episode_source_result(episode)
            for episode in episodes
            if episode.group_id == group_id
        ])[:limit]

        return SourceResults(item_type='fact', item_uuid=uuid, sources=sources)

    if normalized_type == 'entity':
        episodes = await EpisodicNode.get_by_entity_node_uuid(graphiti.driver, uuid)
        sources = sort_episode_sources([
            get_episode_source_result(episode)
            for episode in episodes
            if episode.group_id == group_id
        ])[:limit]

        return SourceResults(item_type='entity', item_uuid=uuid, sources=sources)

    records, _, _ = await graphiti.driver.execute_query(
        """
        MATCH (c:Community {uuid: $uuid, group_id: $group_id})-[:HAS_MEMBER]->(entity:Entity)
        MATCH (episode:Episodic {group_id: $group_id})-[:MENTIONS]->(entity)
        WITH
            episode,
            count(DISTINCT entity) AS matched_entity_count,
            collect(DISTINCT entity.name)[0..5] AS matched_entity_names
        RETURN
            episode.uuid AS uuid,
            episode.name AS name,
            episode.source AS source,
            episode.source_description AS source_description,
            episode.content AS content,
            episode.valid_at AS valid_at,
            episode.created_at AS created_at,
            matched_entity_count,
            matched_entity_names
        ORDER BY matched_entity_count DESC, episode.valid_at DESC, episode.created_at DESC
        LIMIT $limit
        """,
        uuid=uuid,
        group_id=group_id,
        limit=limit,
        routing_='r',
    )

    sources = [
        EpisodeSourceResult(
            uuid=record['uuid'],
            name=record['name'],
            source=record['source'],
            source_description=record['source_description'],
            content=record['content'],
            valid_at=parse_db_date(record['valid_at']),
            created_at=parse_db_date(record['created_at']),
            matched_entity_count=record['matched_entity_count'],
            matched_entity_names=record['matched_entity_names'] or [],
        )
        for record in records
    ]

    return SourceResults(item_type='topic', item_uuid=uuid, sources=sources)


@router.get('/entity-edge/{uuid}', status_code=status.HTTP_200_OK)
async def get_entity_edge(uuid: str, graphiti: ZepGraphitiDep):
    entity_edge = await graphiti.get_entity_edge(uuid)
    return get_fact_result_from_edge(entity_edge)


@router.get('/episodes/{group_id}', status_code=status.HTTP_200_OK)
async def get_episodes(group_id: str, last_n: int, graphiti: ZepGraphitiDep):
    episodes = await graphiti.retrieve_episodes(
        group_ids=[group_id], last_n=last_n, reference_time=datetime.now(timezone.utc)
    )
    return episodes


@router.post('/get-memory', status_code=status.HTTP_200_OK)
async def get_memory(
    request: GetMemoryRequest,
    graphiti: ZepGraphitiDep,
):
    combined_query = compose_query_from_messages(request.messages)
    result = await graphiti.search(
        group_ids=[request.group_id],
        query=combined_query,
        num_results=request.max_facts,
    )
    facts = [get_fact_result_from_edge(edge) for edge in result]
    return GetMemoryResponse(facts=facts)


def compose_query_from_messages(messages: list[Message]):
    combined_query = ''
    for message in messages:
        combined_query += f'{message.role_type or ""}({message.role or ""}): {message.content}\n'
    return combined_query
