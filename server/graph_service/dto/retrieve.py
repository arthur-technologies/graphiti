from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from graph_service.dto.common import Message


class SearchQuery(BaseModel):
    group_ids: list[str] | None = Field(
        None, description='The group ids for the memories to search'
    )
    query: str
    max_facts: int = Field(default=10, description='The maximum number of facts to retrieve')


class ComprehensiveSearchQuery(BaseModel):
    """Query for comprehensive search across all graph layers"""
    group_ids: list[str] | None = Field(
        None, description='The group ids for the memories to search'
    )
    query: str
    max_results: int = Field(default=10, description='Maximum results per category')
    include_facts: bool = Field(default=True, description='Include facts/edges in results')
    include_entities: bool = Field(default=True, description='Include entity nodes in results')
    include_communities: bool = Field(default=True, description='Include community summaries in results')


class FactResult(BaseModel):
    uuid: str
    name: str
    fact: str
    valid_at: datetime | None
    invalid_at: datetime | None
    created_at: datetime
    expired_at: datetime | None

    class Config:
        json_encoders = {datetime: lambda v: v.astimezone(timezone.utc).isoformat()}


class EntityResult(BaseModel):
    """Entity node with summary"""
    uuid: str
    name: str
    entity_type: str | None = None
    summary: str | None = None
    created_at: datetime | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {datetime: lambda v: v.astimezone(timezone.utc).isoformat() if v else None}


class CommunityResult(BaseModel):
    """Community cluster with summary"""
    uuid: str
    name: str
    summary: str | None = None
    created_at: datetime | None = None

    class Config:
        json_encoders = {datetime: lambda v: v.astimezone(timezone.utc).isoformat() if v else None}


class SearchResults(BaseModel):
    facts: list[FactResult]


class ComprehensiveSearchResults(BaseModel):
    """Results from comprehensive search across all graph layers"""
    facts: list[FactResult] = Field(default_factory=list, description='Relationship facts/edges')
    entities: list[EntityResult] = Field(default_factory=list, description='Entity nodes with summaries')
    communities: list[CommunityResult] = Field(default_factory=list, description='Community cluster summaries')


class GetMemoryRequest(BaseModel):
    group_id: str = Field(..., description='The group id of the memory to get')
    max_facts: int = Field(default=10, description='The maximum number of facts to retrieve')
    center_node_uuid: str | None = Field(
        ..., description='The uuid of the node to center the retrieval on'
    )
    messages: list[Message] = Field(
        ..., description='The messages to build the retrieval query from '
    )


class GetMemoryResponse(BaseModel):
    facts: list[FactResult] = Field(..., description='The facts that were retrieved from the graph')
