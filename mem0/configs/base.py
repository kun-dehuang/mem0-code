import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from mem0.embeddings.configs import EmbedderConfig
from mem0.graphs.configs import GraphStoreConfig
from mem0.llms.configs import LlmConfig
from mem0.vector_stores.configs import VectorStoreConfig
from mem0.configs.rerankers.config import RerankerConfig

# Set up the directory path
home_dir = os.path.expanduser("~")
mem0_dir = os.environ.get("MEM0_DIR") or os.path.join(home_dir, ".mem0")


class MemoryItem(BaseModel):
    id: str = Field(..., description="The unique identifier for the text data")
    memory: str = Field(
        ..., description="The memory deduced from the text data"
    )  # TODO After prompt changes from platform, update this
    hash: Optional[str] = Field(None, description="The hash of the memory")
    # The metadata value can be anything and not just string. Fix it
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the text data")
    score: Optional[float] = Field(None, description="The score associated with the text data")
    created_at: Optional[str] = Field(None, description="The timestamp when the memory was created")
    updated_at: Optional[str] = Field(None, description="The timestamp when the memory was updated")


class PromptingConfig(BaseModel):
    source_loader: Optional[str] = Field(default=None, description="Named or importable prompt source loader")
    source_path: Optional[str] = Field(default=None, description="Path to external prompt JSON/YAML file")
    source_format: Optional[str] = Field(default=None, description="Prompt file format override")
    source_config: Dict[str, Any] = Field(default_factory=dict, description="Additional prompt source configuration")
    auto_reload: bool = Field(default=True, description="Reload prompt snapshot before each pipeline run")
    overrides: Dict[str, str] = Field(default_factory=dict, description="Inline prompt template overrides")


class GraphPipelineConfig(BaseModel):
    entity_extractor: str = Field(default="default", description="Graph entity extractor plugin")
    relation_mapper: str = Field(default="default", description="Graph relation mapper plugin")
    entity_resolver: str = Field(default="semantic_similarity", description="Graph entity resolver plugin")
    mutation_planner: str = Field(default="default", description="Graph mutation planner plugin")
    writer: str = Field(default="default", description="Graph writer plugin")


class ObservabilityConfig(BaseModel):
    enable_logger_sink: bool = Field(default=True, description="Emit stage events to logger")
    enable_in_memory_sink: bool = Field(default=False, description="Keep stage events in memory")
    enable_durable_sink: bool = Field(default=False, description="Persist stage events to state store")
    sinks: list[str] = Field(default_factory=list, description="Additional named or importable observer sinks")
    sink_configs: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Observer sink configuration keyed by sink name",
    )


class ConsistencyConfig(BaseModel):
    mode: str = Field(default="recoverable_commit", description="Consistency mode for vector/graph pipeline")


class StateStoreConfig(BaseModel):
    path: Optional[str] = Field(default=None, description="Path to pipeline journal store")


class ProviderRoutingConfig(BaseModel):
    semantic_fact_extraction: Optional[LlmConfig] = Field(default=None)
    graph_entity_extraction: Optional[LlmConfig] = Field(default=None)
    graph_relation_calibration: Optional[LlmConfig] = Field(default=None)
    summary_update_memory: Optional[LlmConfig] = Field(default=None)


class MemoryConfig(BaseModel):
    vector_store: VectorStoreConfig = Field(
        description="Configuration for the vector store",
        default_factory=VectorStoreConfig,
    )
    llm: LlmConfig = Field(
        description="Configuration for the language model",
        default_factory=LlmConfig,
    )
    embedder: EmbedderConfig = Field(
        description="Configuration for the embedding model",
        default_factory=EmbedderConfig,
    )
    history_db_path: str = Field(
        description="Path to the history database",
        default=os.path.join(mem0_dir, "history.db"),
    )
    graph_store: GraphStoreConfig = Field(
        description="Configuration for the graph",
        default_factory=GraphStoreConfig,
    )
    reranker: Optional[RerankerConfig] = Field(
        description="Configuration for the reranker",
        default=None,
    )
    version: str = Field(
        description="The version of the API",
        default="v1.1",
    )
    custom_fact_extraction_prompt: Optional[str] = Field(
        description="Custom prompt for the fact extraction",
        default=None,
    )
    custom_update_memory_prompt: Optional[str] = Field(
        description="Custom prompt for the update memory",
        default=None,
    )
    prompting: PromptingConfig = Field(
        description="Prompt source and override configuration",
        default_factory=PromptingConfig,
    )
    graph_pipeline: GraphPipelineConfig = Field(
        description="Graph pipeline plugin configuration",
        default_factory=GraphPipelineConfig,
    )
    observability: ObservabilityConfig = Field(
        description="Pipeline observability configuration",
        default_factory=ObservabilityConfig,
    )
    consistency: ConsistencyConfig = Field(
        description="Vector/graph consistency configuration",
        default_factory=ConsistencyConfig,
    )
    state_store: StateStoreConfig = Field(
        description="Durable state store configuration",
        default_factory=StateStoreConfig,
    )
    provider_routing: ProviderRoutingConfig = Field(
        description="Stage-specific provider routing",
        default_factory=ProviderRoutingConfig,
    )


class AzureConfig(BaseModel):
    """
    Configuration settings for Azure.

    Args:
        api_key (str): The API key used for authenticating with the Azure service.
        azure_deployment (str): The name of the Azure deployment.
        azure_endpoint (str): The endpoint URL for the Azure service.
        api_version (str): The version of the Azure API being used.
        default_headers (Dict[str, str]): Headers to include in requests to the Azure API.
    """

    api_key: str = Field(
        description="The API key used for authenticating with the Azure service.",
        default=None,
    )
    azure_deployment: str = Field(description="The name of the Azure deployment.", default=None)
    azure_endpoint: str = Field(description="The endpoint URL for the Azure service.", default=None)
    api_version: str = Field(description="The version of the Azure API being used.", default=None)
    default_headers: Optional[Dict[str, str]] = Field(
        description="Headers to include in requests to the Azure API.", default=None
    )
