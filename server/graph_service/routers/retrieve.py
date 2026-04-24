import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, status
from graphiti_core.driver.driver import GraphProvider
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
    GroupStatsResult,
    Message,
    SearchQuery,
    SearchResults,
    SourceResults,
)
from graph_service.zep_graphiti import ZepGraphitiDep, get_fact_result_from_edge

router = APIRouter()

MEETING_SOURCE_DESCRIPTION_PREFIX = 'meeting_transcript:'
ACL_SEARCH_OVERFETCH_MULTIPLIER = 4
ACL_SEARCH_MAX_RESULTS = 50

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


def _parse_count(records: list[dict], key: str = 'count') -> int:
    if not records:
        return 0

    value = records[0].get(key, 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _normalize_allowed_meeting_ids(allowed_meeting_ids: list[str] | None) -> list[str]:
    if not allowed_meeting_ids:
        return []

    return [meeting_id for meeting_id in dict.fromkeys(allowed_meeting_ids) if meeting_id]


def _get_acl_search_limit(requested_limit: int) -> int:
    if requested_limit <= 0:
        return 0

    return min(
        ACL_SEARCH_MAX_RESULTS,
        max(requested_limit, requested_limit * ACL_SEARCH_OVERFETCH_MULTIPLIER),
    )


def _parse_meeting_id_from_source_description(
    source_description: str | None, expected_group_id: str
) -> str | None:
    if not source_description or not source_description.startswith(MEETING_SOURCE_DESCRIPTION_PREFIX):
        return None

    parts = source_description.split(':')
    if len(parts) < 3:
        return None

    _, group_id, meeting_id, *_ = parts
    if group_id != expected_group_id or not meeting_id:
        return None

    return meeting_id


async def _filter_edges_by_allowed_meetings(
    graphiti: ZepGraphitiDep,
    group_id: str,
    edges,
    allowed_meeting_ids: list[str],
):
    if not edges or not allowed_meeting_ids:
        return []

    episode_uuids = list(
        dict.fromkeys(
            episode_uuid
            for edge in edges
            for episode_uuid in (edge.episodes or [])
            if episode_uuid
        )
    )
    if not episode_uuids:
        return []

    allowed_meeting_id_set = set(allowed_meeting_ids)
    episodes = await EpisodicNode.get_by_uuids(graphiti.driver, episode_uuids)
    allowed_episode_uuids = {
        episode.uuid
        for episode in episodes
        if episode.group_id == group_id
        and (
            _parse_meeting_id_from_source_description(
                getattr(episode, 'source_description', None), group_id
            )
            in allowed_meeting_id_set
        )
    }

    if not allowed_episode_uuids:
        return []

    return [
        edge
        for edge in edges
        if any(episode_uuid in allowed_episode_uuids for episode_uuid in (edge.episodes or []))
    ]


async def _get_accessible_entity_uuids(
    graphiti: ZepGraphitiDep,
    group_id: str,
    entity_uuids: list[str],
    allowed_meeting_ids: list[str],
) -> set[str]:
    if not entity_uuids or not allowed_meeting_ids:
        return set()

    records, _, _ = await graphiti.driver.execute_query(
        """
        MATCH (episode:Episodic {group_id: $group_id})-[:MENTIONS]->(entity:Entity {group_id: $group_id})
        WHERE entity.uuid IN $entity_uuids
          AND episode.source_description IS NOT NULL
          AND episode.source_description STARTS WITH $source_prefix
          AND size(split(episode.source_description, ':')) >= 3
          AND split(episode.source_description, ':')[1] = $group_id
          AND split(episode.source_description, ':')[2] IN $allowed_meeting_ids
        RETURN DISTINCT entity.uuid AS uuid
        """,
        group_id=group_id,
        entity_uuids=entity_uuids,
        allowed_meeting_ids=allowed_meeting_ids,
        source_prefix=MEETING_SOURCE_DESCRIPTION_PREFIX,
        routing_='r',
    )

    return {record['uuid'] for record in records if record.get('uuid')}


async def _get_accessible_community_uuids(
    graphiti: ZepGraphitiDep,
    group_id: str,
    community_uuids: list[str],
    allowed_meeting_ids: list[str],
) -> set[str]:
    if not community_uuids or not allowed_meeting_ids:
        return set()

    records, _, _ = await graphiti.driver.execute_query(
        """
        MATCH (community:Community {group_id: $group_id})-[:HAS_MEMBER]->(entity:Entity {group_id: $group_id})
        MATCH (episode:Episodic {group_id: $group_id})-[:MENTIONS]->(entity)
        WHERE community.uuid IN $community_uuids
          AND episode.source_description IS NOT NULL
          AND episode.source_description STARTS WITH $source_prefix
          AND size(split(episode.source_description, ':')) >= 3
          AND split(episode.source_description, ':')[1] = $group_id
          AND split(episode.source_description, ':')[2] IN $allowed_meeting_ids
        RETURN DISTINCT community.uuid AS uuid
        """,
        group_id=group_id,
        community_uuids=community_uuids,
        allowed_meeting_ids=allowed_meeting_ids,
        source_prefix=MEETING_SOURCE_DESCRIPTION_PREFIX,
        routing_='r',
    )

    return {record['uuid'] for record in records if record.get('uuid')}


async def _filter_search_results_by_allowed_meetings(
    graphiti: ZepGraphitiDep,
    group_id: str,
    *,
    edges,
    nodes,
    communities,
    allowed_meeting_ids: list[str],
) -> tuple[list, list, list]:
    if not allowed_meeting_ids:
        return [], [], []

    entity_uuids = [node.uuid for node in nodes]
    community_uuids = [community.uuid for community in communities]

    filtered_edges, accessible_entity_uuids, accessible_community_uuids = await asyncio.gather(
        _filter_edges_by_allowed_meetings(graphiti, group_id, edges, allowed_meeting_ids),
        _get_accessible_entity_uuids(graphiti, group_id, entity_uuids, allowed_meeting_ids),
        _get_accessible_community_uuids(graphiti, group_id, community_uuids, allowed_meeting_ids),
    )

    filtered_nodes = [node for node in nodes if node.uuid in accessible_entity_uuids]
    filtered_communities = [
        community for community in communities if community.uuid in accessible_community_uuids
    ]

    return filtered_edges, filtered_nodes, filtered_communities


def _get_fact_count_query(provider: GraphProvider) -> str:
    if provider == GraphProvider.KUZU:
        return """
            MATCH (n:Entity {group_id: $group_id})-[:RELATES_TO]->(e:RelatesToNode_ {group_id: $group_id})-[:RELATES_TO]->(m:Entity {group_id: $group_id})
            RETURN count(DISTINCT e) AS count
        """

    return """
        MATCH (n:Entity {group_id: $group_id})-[e:RELATES_TO {group_id: $group_id}]->(m:Entity {group_id: $group_id})
        RETURN count(e) AS count
    """


def _get_relationship_type_count_query(provider: GraphProvider) -> str:
    if provider == GraphProvider.KUZU:
        return """
            MATCH (n:Entity {group_id: $group_id})-[:RELATES_TO]->(e:RelatesToNode_ {group_id: $group_id})-[:RELATES_TO]->(:Entity {group_id: $group_id})
            RETURN count(DISTINCT e.name) AS count
        """

    return """
        MATCH (n:Entity {group_id: $group_id})-[e:RELATES_TO {group_id: $group_id}]->(:Entity {group_id: $group_id})
        RETURN count(DISTINCT e.name) AS count
    """


@router.post('/search', status_code=status.HTTP_200_OK)
async def search(query: SearchQuery, graphiti: ZepGraphitiDep):
    allowed_meeting_ids = _normalize_allowed_meeting_ids(query.allowed_meeting_ids)
    requested_limit = max(query.max_facts, 0)
    search_limit = (
        _get_acl_search_limit(requested_limit) if allowed_meeting_ids else requested_limit
    )

    relevant_edges = await graphiti.search(
        group_ids=query.group_ids,
        query=query.query,
        num_results=search_limit,
    )

    if allowed_meeting_ids:
        normalized_group_ids = [group_id for group_id in (query.group_ids or []) if group_id]
        if len(normalized_group_ids) != 1:
            relevant_edges = []
        else:
            relevant_edges = await _filter_edges_by_allowed_meetings(
                graphiti,
                normalized_group_ids[0],
                relevant_edges,
                allowed_meeting_ids,
            )

    facts = [get_fact_result_from_edge(edge) for edge in relevant_edges]
    return SearchResults(
        facts=facts[:requested_limit],
    )


@router.get('/stats/{group_id}', status_code=status.HTTP_200_OK)
async def get_group_stats(group_id: str, graphiti: ZepGraphitiDep):
    driver = graphiti.driver

    entity_records, _, _ = await driver.execute_query(
        """
        MATCH (n:Entity {group_id: $group_id})
        RETURN count(n) AS count
        """,
        group_id=group_id,
        routing_='r',
    )
    community_records, _, _ = await driver.execute_query(
        """
        MATCH (c:Community {group_id: $group_id})
        RETURN count(c) AS count
        """,
        group_id=group_id,
        routing_='r',
    )
    fact_records, _, _ = await driver.execute_query(
        _get_fact_count_query(driver.provider),
        group_id=group_id,
        routing_='r',
    )
    relationship_type_records, _, _ = await driver.execute_query(
        _get_relationship_type_count_query(driver.provider),
        group_id=group_id,
        routing_='r',
    )

    return GroupStatsResult(
        group_id=group_id,
        facts_count=_parse_count(fact_records),
        entities_count=_parse_count(entity_records),
        communities_count=_parse_count(community_records),
        relationship_types_count=_parse_count(relationship_type_records),
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
    allowed_meeting_ids = _normalize_allowed_meeting_ids(query.allowed_meeting_ids)
    requested_limit = max(query.max_results, 0)
    search_limit = (
        _get_acl_search_limit(requested_limit) if allowed_meeting_ids else requested_limit
    )

    # Create config with requested limit
    config = SearchConfig(
        edge_config=MEETING_SEARCH_CONFIG.edge_config if query.include_facts else None,
        node_config=MEETING_SEARCH_CONFIG.node_config if query.include_entities else None,
        community_config=MEETING_SEARCH_CONFIG.community_config if query.include_communities else None,
        limit=search_limit,
        reranker_min_score=MEETING_SEARCH_CONFIG.reranker_min_score,
    )

    # Use search_ for comprehensive results
    results = await graphiti.search_(
        query=query.query,
        config=config,
        group_ids=query.group_ids,
    )

    filtered_edges = results.edges
    filtered_nodes = results.nodes
    filtered_communities = results.communities

    if allowed_meeting_ids:
        normalized_group_ids = [group_id for group_id in (query.group_ids or []) if group_id]
        if len(normalized_group_ids) != 1:
            filtered_edges = []
            filtered_nodes = []
            filtered_communities = []
        else:
            filtered_edges, filtered_nodes, filtered_communities = (
                await _filter_search_results_by_allowed_meetings(
                    graphiti,
                    normalized_group_ids[0],
                    edges=results.edges,
                    nodes=results.nodes,
                    communities=results.communities,
                    allowed_meeting_ids=allowed_meeting_ids,
                )
            )

    # Convert to response format
    facts = (
        [get_fact_result_from_edge(edge) for edge in filtered_edges[:requested_limit]]
        if query.include_facts
        else []
    )
    entities = (
        [get_entity_result(node) for node in filtered_nodes[:requested_limit]]
        if query.include_entities
        else []
    )
    communities = (
        [get_community_result(comm) for comm in filtered_communities[:requested_limit]]
        if query.include_communities
        else []
    )

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
