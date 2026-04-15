"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, Field

from .models import Message, PromptFunction, PromptVersion
from .prompt_helpers import to_prompt_json
from .snippets import summary_instructions


class Summary(BaseModel):
    summary: str = Field(
        ...,
        description='Summary containing the important information about the entity. Under 250 characters',
    )


class SummaryDescription(BaseModel):
    description: str = Field(
        ...,
        description='Short topic-style title for the provided summary, ideally 2 to 6 words',
    )


class Prompt(Protocol):
    summarize_pair: PromptVersion
    summarize_context: PromptVersion
    summary_description: PromptVersion


class Versions(TypedDict):
    summarize_pair: PromptFunction
    summarize_context: PromptFunction
    summary_description: PromptFunction


def summarize_pair(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that combines summaries.',
        ),
        Message(
            role='user',
            content=f"""
        Synthesize the information from the following two summaries into a single succinct summary.

        IMPORTANT: Keep the summary concise and to the point. SUMMARIES MUST BE LESS THAN 250 CHARACTERS.

        Summaries:
        {to_prompt_json(context['node_summaries'])}
        """,
        ),
    ]


def summarize_context(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that generates a summary and attributes from provided text.',
        ),
        Message(
            role='user',
            content=f"""
        Given the MESSAGES and the ENTITY name, create a summary for the ENTITY. Your summary must only use
        information from the provided MESSAGES. Your summary should also only contain information relevant to the
        provided ENTITY.

        In addition, extract any values for the provided entity properties based on their descriptions.
        If the value of the entity property cannot be found in the current context, set the value of the property to the Python value None.

        {summary_instructions}

        <MESSAGES>
        {to_prompt_json(context['previous_episodes'])}
        {to_prompt_json(context['episode_content'])}
        </MESSAGES>

        <ENTITY>
        {context['node_name']}
        </ENTITY>

        <ENTITY CONTEXT>
        {context['node_summary']}
        </ENTITY CONTEXT>

        <ATTRIBUTES>
        {to_prompt_json(context['attributes'])}
        </ATTRIBUTES>
        """,
        ),
    ]


def summary_description(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='You are a helpful assistant that names a cluster of related content with a short professional topic title.',
        ),
        Message(
            role='user',
            content=f"""
        Create a short topic title for the summary below.

        REQUIREMENTS:
        - Return a topic-style label, not a sentence
        - Prefer 2 to 6 words
        - Use Title Case
        - Maximum 60 characters
        - Focus on the main shared subject only
        - Do not start with phrases like "Summary of", "Overview of", "A brief summary of", or "Summarizes"
        - Do not mention that the text is a summary
        - Do not include trailing punctuation
        - Do not include filler words like "discussion of", "topics covered", or "information about"

        GOOD EXAMPLES:
        - Macroeconomic and AI Risk
        - Camera Collectibles and Scams
        - Team Communication Problems

        BAD EXAMPLES:
        - Summary of issues caused by poor coach communication and the need for improved communication
        - A brief summary of claims about a creator community
        - Overview of topics covered: Black comedians' cultural impact and social-commentary humor

        Summary:
        {to_prompt_json(context['summary'])}
        """,
        ),
    ]


versions: Versions = {
    'summarize_pair': summarize_pair,
    'summarize_context': summarize_context,
    'summary_description': summary_description,
}
