from datetime import datetime, timezone

from fastapi import APIRouter, status
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
    EntityResult,
    GetMemoryRequest,
    GetMemoryResponse,
    Message,
    SearchQuery,
    SearchResults,
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
