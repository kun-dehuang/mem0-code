import asyncio
from types import SimpleNamespace

from mem0.memory.main import Memory
from mem0.orchestration import (
    AsyncMemoryRuntime,
    GraphPluginRegistry,
    InMemoryPipelineObserver,
    MemoryRuntime,
    PipelineObserver,
)
from mem0.orchestration.graph_plugins import GraphEntityExtractor, GraphRelationMapper, GraphWriter
from mem0.orchestration.observer import build_pipeline_observer
from mem0.prompting import PromptRegistry
from mem0.storage import SQLitePipelineStateStore


class FakeLLM:
    def __init__(self, provider_name="openai"):
        self.provider_name = provider_name

    def generate_response(self, messages, response_format=None, tools=None):
        if response_format == {"type": "json_object"}:
            return '{"facts":["Alice likes ramen"]}'

        tool_name = tools[0]["function"]["name"] if tools else ""
        if tool_name == "extract_entities":
            return {
                "tool_calls": [
                    {
                        "name": "extract_entities",
                        "arguments": {
                            "entities": [
                                {"entity": "alice", "entity_type": "person"},
                                {"entity": "ramen", "entity_type": "object"},
                            ]
                        },
                    }
                ]
            }
        if tool_name == "establish_relationships":
            return {
                "tool_calls": [
                    {
                        "name": "establish_relationships",
                        "arguments": {
                            "entities": [
                                {"source": "alice", "relationship": "likes", "destination": "ramen"}
                            ]
                        },
                    }
                ]
            }
        if tool_name == "delete_graph_memory":
            return {"tool_calls": []}
        return {}


class FakeGraphClient:
    def __init__(self):
        self.queries = []

    def query(self, statement, params=None):
        self.queries.append((statement, params or {}))
        return []


class FakeGraph:
    def __init__(self, fail_on_add=False):
        self.llm = FakeLLM()
        self.embedding_model = SimpleNamespace(embed=lambda text: [0.1, 0.2, len(text)])
        self.threshold = 0.42
        self.graph = FakeGraphClient()
        self.fail_on_add = fail_on_add
        self.add_calls = []
        self.delete_calls = []

    def _remove_spaces_from_entities(self, entities):
        return entities

    def _search_graph_db(self, node_list, filters):
        return []

    def _search_source_node(self, embedding, filters, threshold):
        return []

    def _search_destination_node(self, embedding, filters, threshold):
        return []

    def _delete_entities(self, entities, filters):
        self.delete_calls.append((entities, filters))
        return entities

    def _add_entities(self, entities, filters, entity_type_map):
        if self.fail_on_add:
            raise RuntimeError("Remote Compact Task failed")
        self.add_calls.append((entities, filters, entity_type_map))
        return entities


class FakeMemory:
    def __init__(self, graph, graph_pipeline=None):
        self.graph = graph
        self.enable_graph = True
        self.config = SimpleNamespace(
            llm=SimpleNamespace(provider="openai", config={}),
            graph_store=SimpleNamespace(threshold=0.42, custom_prompt=None),
            graph_pipeline=graph_pipeline or SimpleNamespace(
                entity_extractor="default",
                relation_mapper="default",
                entity_resolver="semantic_similarity",
                mutation_planner="default",
                writer="default",
            ),
            provider_routing=SimpleNamespace(
                semantic_fact_extraction=None,
                graph_entity_extraction=None,
                graph_relation_calibration=None,
                summary_update_memory=None,
            ),
        )

    def _create_llm(self, provider_name, config=None):
        return FakeLLM(provider_name=provider_name)

    def _add_to_vector_store(self, messages, metadata, filters, infer):
        return [{"id": "vec-1", "memory": messages[0]["content"], "event": "ADD", "job_id": metadata["job_id"]}]

    def _add_to_graph(self, messages, filters):
        return [{"fallback": True}]


def _runtime(tmp_path, graph):
    state_store = SQLitePipelineStateStore(str(tmp_path / "pipeline.db"))
    observer = InMemoryPipelineObserver()
    runtime = MemoryRuntime(
        memory=FakeMemory(graph),
        state_store=state_store,
        prompt_registry=PromptRegistry(),
        observer=observer,
    )
    return runtime, state_store, observer


def test_runtime_persists_pipeline_and_provenance(tmp_path):
    runtime, state_store, observer = _runtime(tmp_path, FakeGraph())
    metadata = {"analysis_batch_id": "batch-1", "source_ref": "log://1"}
    result = runtime.add(
        [{"role": "user", "content": "Alice likes ramen"}],
        metadata=metadata,
        filters={"user_id": "alice"},
        infer=True,
    )

    job = state_store.get_job(metadata["job_id"])
    assert result["results"][0]["id"] == "vec-1"
    assert job is not None
    assert job["phase"] == "confirmed"
    assert job["payload"]["graph_plan"]["relations_to_add"] == [
        {"source": "alice", "relationship": "likes", "destination": "ramen"}
    ]
    assert any(event["stage_name"] == "GraphMutationPlanningStage" for event in observer.events)
    assert runtime.memory.graph.graph.queries


def test_runtime_replays_graph_after_writer_failure(tmp_path):
    failing_graph = FakeGraph(fail_on_add=True)
    runtime, state_store, _ = _runtime(tmp_path, failing_graph)
    metadata = {"analysis_batch_id": "batch-2", "source_ref": "log://2"}

    try:
        runtime.add(
            [{"role": "user", "content": "Alice likes ramen"}],
            metadata=metadata,
            filters={"user_id": "alice"},
            infer=True,
        )
    except RuntimeError as exc:
        assert "Remote Compact Task failed" in str(exc)
    else:
        raise AssertionError("expected graph write failure")

    job_id = metadata["job_id"]
    failed_job = state_store.get_job(job_id)
    assert failed_job is not None
    assert failed_job["phase"] == "vector_applied"
    assert failed_job["payload"]["graph_plan"]["relations_to_add"]

    runtime.memory.graph.fail_on_add = False
    replay_result = runtime.replay_graph(job_id)
    recovered_job = state_store.get_job(job_id)

    assert replay_result is not None
    assert replay_result["added_entities"] == [
        {"source": "alice", "relationship": "likes", "destination": "ramen"}
    ]
    assert recovered_job is not None
    assert recovered_job["phase"] == "confirmed"


class CustomEntityExtractor(GraphEntityExtractor):
    def extract(self, graph, llm, llm_provider, prompt_registry, context):
        return {"custom_entity": "person"}


class CustomWriter(GraphWriter):
    def apply(self, graph, plan, filters, context):
        return {"deleted_entities": [], "added_entities": [{"custom": True, "job_id": context["job_id"]}]}


def test_runtime_uses_registered_graph_plugins(tmp_path):
    registry = GraphPluginRegistry()
    registry.register_entity_extractor("custom_extractor", CustomEntityExtractor)
    registry.register_writer("custom_writer", CustomWriter)

    graph_pipeline = SimpleNamespace(
        entity_extractor="custom_extractor",
        relation_mapper="default",
        entity_resolver="semantic_similarity",
        mutation_planner="default",
        writer="custom_writer",
    )
    memory = FakeMemory(FakeGraph(), graph_pipeline=graph_pipeline)
    state_store = SQLitePipelineStateStore(str(tmp_path / "pipeline.db"))
    observer = InMemoryPipelineObserver()
    runtime = MemoryRuntime(
        memory=memory,
        state_store=state_store,
        prompt_registry=PromptRegistry(),
        observer=observer,
        graph_plugin_registry=registry,
    )

    metadata = {"source_ref": "log://3"}
    result = runtime.add(
        [{"role": "user", "content": "Alice likes ramen"}],
        metadata=metadata,
        filters={"user_id": "alice"},
        infer=True,
    )

    assert result["relations"]["added_entities"] == [{"custom": True, "job_id": metadata["job_id"]}]


class RecordingObserver(PipelineObserver):
    def __init__(self):
        self.events = []

    def emit(self, job_id, stage_name, event_type, payload=None):
        self.events.append((job_id, stage_name, event_type, payload or {}))


def build_recording_observer(state_store, sink_config):
    observer = RecordingObserver()
    observer.sink_name = sink_config.get("name", "recording")
    return observer


def test_memory_registration_helpers_register_plugins_and_observers(tmp_path):
    Memory.register_graph_entity_extractor("custom_extractor_via_memory", CustomEntityExtractor)
    Memory.register_graph_writer("custom_writer_via_memory", CustomWriter)
    Memory.register_pipeline_observer("recording", build_recording_observer)

    graph_pipeline = SimpleNamespace(
        entity_extractor="custom_extractor_via_memory",
        relation_mapper="default",
        entity_resolver="semantic_similarity",
        mutation_planner="default",
        writer="custom_writer_via_memory",
    )
    memory = FakeMemory(FakeGraph(), graph_pipeline=graph_pipeline)
    state_store = SQLitePipelineStateStore(str(tmp_path / "pipeline.db"))
    config = SimpleNamespace(
        observability=SimpleNamespace(
            enable_logger_sink=False,
            enable_in_memory_sink=False,
            enable_durable_sink=False,
            sinks=["recording"],
            sink_configs={"recording": {"name": "sdk-registered"}},
        )
    )
    observer = build_pipeline_observer(config, state_store)
    runtime = MemoryRuntime(
        memory=memory,
        state_store=state_store,
        prompt_registry=PromptRegistry(),
        observer=observer,
    )

    result = runtime.add(
        [{"role": "user", "content": "Alice likes ramen"}],
        metadata={"source_ref": "log://4"},
        filters={"user_id": "alice"},
        infer=True,
    )

    recording_observers = [item for item in observer.observers if isinstance(item, RecordingObserver)]
    assert result["relations"]["added_entities"][0]["custom"] is True
    assert len(recording_observers) == 1
    assert recording_observers[0].sink_name == "sdk-registered"
    assert recording_observers[0].events


class LegacyGraphBackend:
    pass


def test_runtime_falls_back_to_legacy_graph_backend(tmp_path):
    state_store = SQLitePipelineStateStore(str(tmp_path / "pipeline.db"))
    observer = InMemoryPipelineObserver()
    memory = FakeMemory(LegacyGraphBackend())
    runtime = MemoryRuntime(
        memory=memory,
        state_store=state_store,
        prompt_registry=PromptRegistry(),
        observer=observer,
    )

    result = runtime.add(
        [{"role": "user", "content": "Alice likes ramen"}],
        metadata={"source_ref": "log://5"},
        filters={"user_id": "alice"},
        infer=True,
    )

    assert result["relations"] == [{"fallback": True}]
    assert any(event["stage_name"] == "GraphCompatibilityStage" for event in observer.events)


class TrackingEntityExtractor(GraphEntityExtractor):
    seen_provider_names = []

    def extract(self, graph, llm, llm_provider, prompt_registry, context):
        self.__class__.seen_provider_names.append((llm_provider, getattr(llm, "provider_name", None)))
        return {"alice": "person"}


class TrackingRelationMapper(GraphRelationMapper):
    seen_provider_names = []

    def map(self, graph, llm, llm_provider, prompt_registry, context):
        self.__class__.seen_provider_names.append((llm_provider, getattr(llm, "provider_name", None)))
        return [{"source": "alice", "relationship": "likes", "destination": "ramen"}]


def test_runtime_uses_stage_provider_routing_for_graph_plugins(tmp_path):
    TrackingEntityExtractor.seen_provider_names = []
    TrackingRelationMapper.seen_provider_names = []

    registry = GraphPluginRegistry()
    registry.register_entity_extractor("tracking_extractor", TrackingEntityExtractor)
    registry.register_relation_mapper("tracking_mapper", TrackingRelationMapper)

    graph_pipeline = SimpleNamespace(
        entity_extractor="tracking_extractor",
        relation_mapper="tracking_mapper",
        entity_resolver="semantic_similarity",
        mutation_planner="default",
        writer="default",
    )
    memory = FakeMemory(FakeGraph(), graph_pipeline=graph_pipeline)
    memory.config.provider_routing = SimpleNamespace(
        semantic_fact_extraction=None,
        graph_entity_extraction=SimpleNamespace(provider="gemini", config={"model": "gemini-pro"}),
        graph_relation_calibration=SimpleNamespace(provider="claude", config={"model": "sonnet"}),
        summary_update_memory=None,
    )
    runtime = MemoryRuntime(
        memory=memory,
        state_store=SQLitePipelineStateStore(str(tmp_path / "pipeline.db")),
        prompt_registry=PromptRegistry(),
        observer=InMemoryPipelineObserver(),
        graph_plugin_registry=registry,
    )

    runtime.add(
        [{"role": "user", "content": "Alice likes ramen"}],
        metadata={"source_ref": "log://6"},
        filters={"user_id": "alice"},
        infer=True,
    )

    assert TrackingEntityExtractor.seen_provider_names == [("gemini", "gemini")]
    assert TrackingRelationMapper.seen_provider_names == [("claude", "claude")]


class AsyncFakeMemory(FakeMemory):
    async def _add_to_vector_store(self, messages, metadata, filters, infer):
        await asyncio.sleep(0)
        return super()._add_to_vector_store(messages, metadata, filters, infer)


def test_async_runtime_persists_pipeline(tmp_path):
    state_store = SQLitePipelineStateStore(str(tmp_path / "pipeline.db"))
    observer = InMemoryPipelineObserver()
    runtime = AsyncMemoryRuntime(
        memory=AsyncFakeMemory(FakeGraph()),
        state_store=state_store,
        prompt_registry=PromptRegistry(),
        observer=observer,
    )
    metadata = {"source_ref": "log://7"}

    result = asyncio.run(
        runtime.add(
            [{"role": "user", "content": "Alice likes ramen"}],
            metadata=metadata,
            filters={"user_id": "alice"},
            infer=True,
        )
    )

    job = state_store.get_job(metadata["job_id"])
    assert result["results"][0]["id"] == "vec-1"
    assert job is not None
    assert job["phase"] == "confirmed"


def test_build_pipeline_observer_writes_durable_events(tmp_path):
    state_store = SQLitePipelineStateStore(str(tmp_path / "pipeline.db"))
    config = SimpleNamespace(
        observability=SimpleNamespace(
            enable_logger_sink=False,
            enable_in_memory_sink=False,
            enable_durable_sink=True,
            sinks=[],
            sink_configs={},
        )
    )
    observer = build_pipeline_observer(config, state_store)
    observer.emit("job-1", "stage-1", "completed", {"value": 1})

    events = state_store.list_events("job-1")
    assert events == [
        {
            "stage_name": "stage-1",
            "event_type": "completed",
            "payload": {"value": 1},
            "created_at": events[0]["created_at"],
        }
    ]
