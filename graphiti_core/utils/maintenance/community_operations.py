import asyncio
import logging
import os
import re
from collections import defaultdict
from time import perf_counter

from pydantic import BaseModel

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.edges import CommunityEdge
from graphiti_core.embedder import EmbedderClient
from graphiti_core.helpers import semaphore_gather
from graphiti_core.llm_client import LLMClient
from graphiti_core.models.nodes.node_db_queries import COMMUNITY_NODE_RETURN
from graphiti_core.nodes import CommunityNode, EntityNode, get_community_node_from_record
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.summarize_nodes import Summary, SummaryDescription
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.edge_operations import build_community_edges

MAX_COMMUNITY_BUILD_CONCURRENCY = max(
    1,
    int(os.getenv('GRAPHITI_COMMUNITY_BUILD_CONCURRENCY', '10')),
)
MAX_LABEL_PROPAGATION_ITERATIONS = max(
    25,
    int(os.getenv('GRAPHITI_LABEL_PROPAGATION_MAX_ITERATIONS', '500')),
)
LABEL_PROPAGATION_LOG_EVERY = max(
    5,
    int(os.getenv('GRAPHITI_LABEL_PROPAGATION_LOG_EVERY', '25')),
)
MIN_COMMUNITY_CLUSTER_SIZE = max(
    2,
    int(os.getenv('GRAPHITI_COMMUNITY_MIN_CLUSTER_SIZE', '2')),
)
MIN_EMERGING_COMMUNITY_CLUSTER_SIZE = max(
    MIN_COMMUNITY_CLUSTER_SIZE,
    int(
        os.getenv(
            'GRAPHITI_EMERGING_COMMUNITY_MIN_CLUSTER_SIZE',
            str(max(3, MIN_COMMUNITY_CLUSTER_SIZE)),
        )
    ),
)

logger = logging.getLogger(__name__)


def emit_progress(message: str, *args: object) -> None:
    if args:
        message = message % args
    logger.info(message)
    print(message, flush=True)


TOPIC_NAME_MAX_LENGTH = max(
    24,
    int(os.getenv('GRAPHITI_COMMUNITY_NAME_MAX_LENGTH', '60')),
)


def normalize_community_name(name: str) -> str:
    value = re.sub(r'\s+', ' ', name).strip()
    value = value.strip(' \'"“”‘’.,;:!?-')

    noisy_prefixes = (
        'summary of ',
        'overview of ',
        'a brief summary of ',
        'a concise comparison summarizing ',
        'a concise summary of ',
        'summarizes ',
        'summary describes ',
        'discussion of ',
        'topics covered: ',
        'information about ',
    )
    lower_value = value.lower()
    for prefix in noisy_prefixes:
        if lower_value.startswith(prefix):
            value = value[len(prefix):].strip()
            lower_value = value.lower()
            break

    if ':' in value:
        value = value.split(':', 1)[-1].strip()

    if len(value) > TOPIC_NAME_MAX_LENGTH:
        truncated = value[:TOPIC_NAME_MAX_LENGTH].rsplit(' ', 1)[0].strip()
        value = truncated or value[:TOPIC_NAME_MAX_LENGTH].strip()

    return value.strip(' \'"“”‘’.,;:!?-') or 'Related Topics'


class Neighbor(BaseModel):
    node_uuid: str
    edge_count: int


async def get_community_clusters(
    driver: GraphDriver, group_ids: list[str] | None
) -> list[list[EntityNode]]:
    if driver.graph_operations_interface:
        try:
            return await driver.graph_operations_interface.get_community_clusters(
                driver, group_ids
            )
        except NotImplementedError:
            pass

    community_clusters: list[list[EntityNode]] = []

    if group_ids is None:
        group_id_values, _, _ = await driver.execute_query(
            """
            MATCH (n:Entity)
            WHERE n.group_id IS NOT NULL
            RETURN
                collect(DISTINCT n.group_id) AS group_ids
            """
        )

        group_ids = group_id_values[0]['group_ids'] if group_id_values else []

    for group_id in group_ids:
        group_start = perf_counter()
        projection: dict[str, list[Neighbor]] = {}
        nodes = await EntityNode.get_by_group_ids(driver, [group_id])
        emit_progress(
            'Community clustering: group %s has %s entity nodes before label propagation',
            group_id,
            len(nodes),
        )
        for node in nodes:
            match_query = """
                MATCH (n:Entity {group_id: $group_id, uuid: $uuid})-[e:RELATES_TO]-(m: Entity {group_id: $group_id})
            """
            if driver.provider == GraphProvider.KUZU:
                match_query = """
                MATCH (n:Entity {group_id: $group_id, uuid: $uuid})-[:RELATES_TO]-(e:RelatesToNode_)-[:RELATES_TO]-(m: Entity {group_id: $group_id})
                """
            records, _, _ = await driver.execute_query(
                match_query
                + """
                WITH count(e) AS count, m.uuid AS uuid
                RETURN
                    uuid,
                    count
                """,
                uuid=node.uuid,
                group_id=group_id,
            )

            projection[node.uuid] = [
                Neighbor(node_uuid=record['uuid'], edge_count=record['count']) for record in records
            ]

        cluster_uuids = label_propagation(projection)
        emit_progress(
            'Community clustering: group %s produced %s raw clusters in %.2fs',
            group_id,
            len(cluster_uuids),
            perf_counter() - group_start,
        )

        community_clusters.extend(
            list(
                await semaphore_gather(
                    *[EntityNode.get_by_uuids(driver, cluster) for cluster in cluster_uuids]
                )
            )
        )

    return community_clusters


def label_propagation(projection: dict[str, list[Neighbor]]) -> list[list[str]]:
    # Implement the label propagation community detection algorithm.
    # 1. Start with each node being assigned its own community
    # 2. Each node will take on the community of the plurality of its neighbors
    # 3. Ties are broken by going to the largest community
    # 4. Continue until no communities change during propagation

    community_map = {uuid: i for i, uuid in enumerate(projection.keys())}
    node_order = list(projection.keys())
    seen_states = {tuple(community_map[uuid] for uuid in node_order)}

    for iteration in range(1, MAX_LABEL_PROPAGATION_ITERATIONS + 1):
        changed_nodes = 0

        # Update labels in-place so later nodes can see the most recent community
        # assignments from earlier nodes in the same iteration.
        for uuid in node_order:
            neighbors = projection[uuid]
            curr_community = community_map[uuid]

            community_candidates: dict[int, int] = defaultdict(int)
            for neighbor in neighbors:
                community_candidates[community_map[neighbor.node_uuid]] += neighbor.edge_count
            community_lst = [
                (count, community) for community, count in community_candidates.items()
            ]

            community_lst.sort(reverse=True)
            candidate_rank, community_candidate = community_lst[0] if community_lst else (0, -1)
            if community_candidate != -1 and candidate_rank > 1:
                new_community = community_candidate
            else:
                new_community = max(community_candidate, curr_community)

            if new_community != curr_community:
                community_map[uuid] = new_community
                changed_nodes += 1

        state_signature = tuple(community_map[uuid] for uuid in node_order)
        community_count = len(set(state_signature))

        if (
            iteration <= 3
            or iteration % LABEL_PROPAGATION_LOG_EVERY == 0
            or changed_nodes == 0
        ):
            emit_progress(
                'Community clustering: label propagation iteration %s changed %s nodes (communities=%s)',
                iteration,
                changed_nodes,
                community_count,
            )

        if changed_nodes == 0:
            emit_progress(
                'Community clustering: label propagation converged after %s iterations',
                iteration,
            )
            break

        if state_signature in seen_states:
            emit_progress(
                'Community clustering: label propagation detected oscillation after %s iterations; using current partition with %s communities',
                iteration,
                community_count,
            )
            break

        seen_states.add(state_signature)
    else:
        emit_progress(
            'Community clustering: label propagation hit max iterations (%s); using current partition with %s communities',
            MAX_LABEL_PROPAGATION_ITERATIONS,
            len(set(community_map.values())),
        )

    community_cluster_map = defaultdict(list)
    for uuid, community in community_map.items():
        community_cluster_map[community].append(uuid)

    clusters = [cluster for cluster in community_cluster_map.values()]
    return clusters


async def summarize_pair(llm_client: LLMClient, summary_pair: tuple[str, str]) -> str:
    # Prepare context for LLM
    context = {
        'node_summaries': [{'summary': summary} for summary in summary_pair],
    }

    llm_response = await llm_client.generate_response(
        prompt_library.summarize_nodes.summarize_pair(context),
        response_model=Summary,
        prompt_name='summarize_nodes.summarize_pair',
    )

    pair_summary = llm_response.get('summary', '')

    return pair_summary


async def generate_summary_description(llm_client: LLMClient, summary: str) -> str:
    context = {
        'summary': summary,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.summarize_nodes.summary_description(context),
        response_model=SummaryDescription,
        prompt_name='summarize_nodes.summary_description',
    )

    description = normalize_community_name(llm_response.get('description', ''))

    return description


async def summarize_summaries(llm_client: LLMClient, summaries: list[str]) -> str:
    filtered_summaries = [summary.strip() for summary in summaries if summary and summary.strip()]

    if not filtered_summaries:
        return ''

    if len(filtered_summaries) == 1:
        return filtered_summaries[0]

    current_summaries = filtered_summaries
    length = len(current_summaries)
    while length > 1:
        odd_one_out: str | None = None
        if length % 2 == 1:
            odd_one_out = current_summaries.pop()
            length -= 1
        new_summaries: list[str] = list(
            await semaphore_gather(
                *[
                    summarize_pair(llm_client, (str(left_summary), str(right_summary)))
                    for left_summary, right_summary in zip(
                        current_summaries[: int(length / 2)],
                        current_summaries[int(length / 2) :],
                        strict=False,
                    )
                ]
            )
        )
        if odd_one_out is not None:
            new_summaries.append(odd_one_out)
        current_summaries = new_summaries
        length = len(current_summaries)

    return current_summaries[0]


async def build_community(
    llm_client: LLMClient, community_cluster: list[EntityNode]
) -> tuple[CommunityNode, list[CommunityEdge]]:
    community_start = perf_counter()
    summary = await summarize_summaries(
        llm_client, [entity.summary for entity in community_cluster]
    )
    name = await generate_summary_description(llm_client, summary)
    now = utc_now()
    community_node = CommunityNode(
        name=name,
        group_id=community_cluster[0].group_id,
        labels=['Community'],
        created_at=now,
        summary=summary,
    )
    community_edges = build_community_edges(community_cluster, community_node, now)

    logger.debug((community_node, community_edges))
    emit_progress(
        'Community build: cluster size %s produced "%s" in %.2fs',
        len(community_cluster),
        community_node.name,
        perf_counter() - community_start,
    )

    return community_node, community_edges


async def build_communities(
    driver: GraphDriver,
    llm_client: LLMClient,
    group_ids: list[str] | None,
) -> tuple[list[CommunityNode], list[CommunityEdge]]:
    build_start = perf_counter()
    community_clusters = await get_community_clusters(driver, group_ids)
    filtered_clusters = [
        cluster for cluster in community_clusters if len(cluster) >= MIN_COMMUNITY_CLUSTER_SIZE
    ]

    skipped_clusters = len(community_clusters) - len(filtered_clusters)
    emit_progress(
        'Community build candidate clusters: %s kept, %s skipped below min size %s',
        len(filtered_clusters),
        skipped_clusters,
        MIN_COMMUNITY_CLUSTER_SIZE,
    )
    emit_progress(
        'Community build configuration: concurrency=%s, emerging_min_cluster_size=%s',
        MAX_COMMUNITY_BUILD_CONCURRENCY,
        MIN_EMERGING_COMMUNITY_CLUSTER_SIZE,
    )

    if not filtered_clusters:
        return [], []

    semaphore = asyncio.Semaphore(MAX_COMMUNITY_BUILD_CONCURRENCY)

    async def limited_build_community(cluster: list[EntityNode], cluster_index: int):
        async with semaphore:
            if cluster_index < 5 or (cluster_index + 1) % 10 == 0:
                emit_progress(
                    'Community build progress: starting cluster %s/%s (size=%s)',
                    cluster_index + 1,
                    len(filtered_clusters),
                    len(cluster),
                )
            return await build_community(llm_client, cluster)

    communities: list[tuple[CommunityNode, list[CommunityEdge]]] = list(
        await semaphore_gather(
            *[
                limited_build_community(cluster, cluster_index)
                for cluster_index, cluster in enumerate(filtered_clusters)
            ]
        )
    )

    community_nodes: list[CommunityNode] = []
    community_edges: list[CommunityEdge] = []
    for community in communities:
        community_nodes.append(community[0])
        community_edges.extend(community[1])

    emit_progress(
        'Community build finished: %s community nodes, %s community edges in %.2fs',
        len(community_nodes),
        len(community_edges),
        perf_counter() - build_start,
    )

    return community_nodes, community_edges


async def remove_communities(
    driver: GraphDriver, group_ids: list[str] | None = None
):
    if driver.graph_operations_interface:
        try:
            return await driver.graph_operations_interface.remove_communities(driver)
        except NotImplementedError:
            pass

    if group_ids:
        await driver.execute_query(
            """
            MATCH (c:Community)
            WHERE c.group_id IN $group_ids
            DETACH DELETE c
            """,
            group_ids=group_ids,
        )
        return

    await driver.execute_query(
        """
        MATCH (c:Community)
        DETACH DELETE c
        """
    )


async def determine_entity_community(
    driver: GraphDriver, entity: EntityNode
) -> tuple[CommunityNode | None, bool]:
    if driver.graph_operations_interface:
        try:
            return await driver.graph_operations_interface.determine_entity_community(
                driver, entity
            )
        except NotImplementedError:
            pass

    # Check if the node is already part of a community
    records, _, _ = await driver.execute_query(
        """
        MATCH (c:Community)-[:HAS_MEMBER]->(n:Entity {uuid: $entity_uuid})
        RETURN
        """
        + COMMUNITY_NODE_RETURN,
        entity_uuid=entity.uuid,
    )

    if len(records) > 0:
        return get_community_node_from_record(records[0]), False

    # If the node has no community, add it to the mode community of surrounding entities
    match_query = """
        MATCH (c:Community)-[:HAS_MEMBER]->(m:Entity)-[:RELATES_TO]-(n:Entity {uuid: $entity_uuid})
    """
    if driver.provider == GraphProvider.KUZU:
        match_query = """
            MATCH (c:Community)-[:HAS_MEMBER]->(m:Entity)-[:RELATES_TO]-(e:RelatesToNode_)-[:RELATES_TO]-(n:Entity {uuid: $entity_uuid})
        """
    records, _, _ = await driver.execute_query(
        match_query
        + """
        RETURN
        """
        + COMMUNITY_NODE_RETURN,
        entity_uuid=entity.uuid,
    )

    communities: list[CommunityNode] = [
        get_community_node_from_record(record) for record in records
    ]

    community_map: dict[str, int] = defaultdict(int)
    for community in communities:
        community_map[community.uuid] += 1

    community_uuid = None
    max_count = 0
    for uuid, count in community_map.items():
        if count > max_count:
            community_uuid = uuid
            max_count = count

    if max_count == 0:
        return None, False

    for community in communities:
        if community.uuid == community_uuid:
            return community, True

    return None, False


async def update_community(
    driver: GraphDriver,
    llm_client: LLMClient,
    embedder: EmbedderClient,
    entity: EntityNode,
) -> tuple[list[CommunityNode], list[CommunityEdge]]:
    community, is_new = await determine_entity_community(driver, entity)

    if community is None:
        return [], []

    new_summary = await summarize_pair(llm_client, (entity.summary, community.summary))
    new_name = await generate_summary_description(llm_client, new_summary)

    community.summary = new_summary
    community.name = new_name

    community_edges = []
    if is_new:
        community_edge = (build_community_edges([entity], community, utc_now()))[0]
        await community_edge.save(driver)
        community_edges.append(community_edge)

    await community.generate_name_embedding(embedder)

    await community.save(driver)

    return [community], community_edges


async def refresh_community(
    driver: GraphDriver,
    llm_client: LLMClient,
    embedder: EmbedderClient,
    community: CommunityNode,
    touched_entities: list[EntityNode],
    new_member_entities: list[EntityNode],
) -> tuple[CommunityNode, list[CommunityEdge]]:
    summary_inputs = [community.summary, *[entity.summary for entity in touched_entities]]
    new_summary = await summarize_summaries(llm_client, summary_inputs)
    new_name = await generate_summary_description(llm_client, new_summary)

    community.summary = new_summary
    community.name = new_name

    community_edges: list[CommunityEdge] = []
    if new_member_entities:
        new_edges = build_community_edges(new_member_entities, community, utc_now())
        for edge in new_edges:
            await edge.save(driver)
        community_edges.extend(new_edges)

    await community.generate_name_embedding(embedder)
    await community.save(driver)

    return community, community_edges


async def get_subset_community_clusters(
    driver: GraphDriver,
    group_id: str,
    entity_uuids: list[str],
) -> list[list[EntityNode]]:
    unique_entity_uuids = list(dict.fromkeys(entity_uuids))
    if not unique_entity_uuids:
        return []

    projection: dict[str, list[Neighbor]] = {}
    for entity_uuid in unique_entity_uuids:
        match_query = """
            MATCH (n:Entity {group_id: $group_id, uuid: $uuid})-[e:RELATES_TO]-(m: Entity {group_id: $group_id})
            WHERE m.uuid IN $entity_uuids
        """
        if driver.provider == GraphProvider.KUZU:
            match_query = """
                MATCH (n:Entity {group_id: $group_id, uuid: $uuid})-[:RELATES_TO]-(e:RelatesToNode_)-[:RELATES_TO]-(m: Entity {group_id: $group_id})
                WHERE m.uuid IN $entity_uuids
            """
        records, _, _ = await driver.execute_query(
            match_query
            + """
            WITH count(e) AS count, m.uuid AS uuid
            RETURN
                uuid,
                count
            """,
            uuid=entity_uuid,
            group_id=group_id,
            entity_uuids=unique_entity_uuids,
        )

        projection[entity_uuid] = [
            Neighbor(node_uuid=record['uuid'], edge_count=record['count']) for record in records
        ]

    cluster_uuids = label_propagation(projection)
    filtered_cluster_uuids = [
        cluster for cluster in cluster_uuids if len(cluster) >= MIN_EMERGING_COMMUNITY_CLUSTER_SIZE
    ]

    if not filtered_cluster_uuids:
        return []

    return list(
        await semaphore_gather(
            *[EntityNode.get_by_uuids(driver, cluster) for cluster in filtered_cluster_uuids]
        )
    )


async def maintain_communities_for_entities(
    driver: GraphDriver,
    llm_client: LLMClient,
    embedder: EmbedderClient,
    group_id: str,
    entity_uuids: list[str],
) -> dict[str, object]:
    unique_entity_uuids = list(dict.fromkeys(entity_uuids))
    if not unique_entity_uuids:
        return {
            'communities_updated': 0,
            'communities_created': 0,
            'updated_community_names': [],
            'created_community_names': [],
            'unassigned_entity_uuids': [],
        }

    touched_entities = await EntityNode.get_by_uuids(driver, unique_entity_uuids)
    entity_map = {entity.uuid: entity for entity in touched_entities}

    assigned_communities: dict[
        str, dict[str, CommunityNode | list[EntityNode]]
    ] = {}
    unassigned_entity_uuids: list[str] = []

    for entity_uuid in unique_entity_uuids:
        entity = entity_map.get(entity_uuid)
        if entity is None:
            continue

        community, is_new = await determine_entity_community(driver, entity)
        if community is None:
            unassigned_entity_uuids.append(entity.uuid)
            continue

        community_bucket = assigned_communities.setdefault(
            community.uuid,
            {
                'community': community,
                'touched_entities': [],
                'new_member_entities': [],
            },
        )
        touched_entities_bucket = community_bucket['touched_entities']
        if isinstance(touched_entities_bucket, list):
            touched_entities_bucket.append(entity)
        if is_new:
            new_member_entities_bucket = community_bucket['new_member_entities']
            if isinstance(new_member_entities_bucket, list):
                new_member_entities_bucket.append(entity)

    refreshed_communities: list[CommunityNode] = []
    refreshed_edges: list[CommunityEdge] = []
    for community_data in assigned_communities.values():
        community = community_data['community']
        touched_entity_list = community_data['touched_entities']
        new_member_entity_list = community_data['new_member_entities']
        if not isinstance(community, CommunityNode):
            continue
        if not isinstance(touched_entity_list, list):
            touched_entity_list = []
        if not isinstance(new_member_entity_list, list):
            new_member_entity_list = []

        refreshed_community, new_edges = await refresh_community(
            driver,
            llm_client,
            embedder,
            community,
            touched_entity_list,
            new_member_entity_list,
        )
        refreshed_communities.append(refreshed_community)
        refreshed_edges.extend(new_edges)

    emerging_clusters = await get_subset_community_clusters(
        driver, group_id, unassigned_entity_uuids
    )
    emerging_communities: list[CommunityNode] = []
    emerging_edges: list[CommunityEdge] = []
    for cluster in emerging_clusters:
        community_node, community_edges = await build_community(llm_client, cluster)
        await community_node.generate_name_embedding(embedder)
        await community_node.save(driver)
        for edge in community_edges:
            await edge.save(driver)
        emerging_communities.append(community_node)
        emerging_edges.extend(community_edges)

    emit_progress(
        'Community maintenance for group %s: %s updated, %s created, %s unassigned entities',
        group_id,
        len(refreshed_communities),
        len(emerging_communities),
        len(unassigned_entity_uuids),
    )

    return {
        'communities_updated': len(refreshed_communities),
        'communities_created': len(emerging_communities),
        'updated_community_names': [community.name for community in refreshed_communities],
        'created_community_names': [community.name for community in emerging_communities],
        'unassigned_entity_uuids': unassigned_entity_uuids,
        'community_edges_created': len(refreshed_edges) + len(emerging_edges),
    }
