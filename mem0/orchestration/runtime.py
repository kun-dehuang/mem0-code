from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from mem0.orchestration.graph_plugins import (
    DEFAULT_GRAPH_PLUGIN_REGISTRY,
    GraphPluginRegistry,
    resolve_graph_plugins,
)


def _supports_modular_graph_pipeline(graph: Any) -> bool:
    graph_type = type(graph)
    if graph_type.__module__.startswith("unittest.mock"):
        return False

    required_attrs = (
        "llm",
        "embedding_model",
        "_remove_spaces_from_entities",
        "_search_graph_db",
        "_delete_entities",
        "_add_entities",
        "_search_source_node",
        "_search_destination_node",
    )
    return all(hasattr(graph, attr) for attr in required_attrs)


@dataclass
class CommitCoordinator:
    state_store: Any
    observer: Any

    def prepare(self, job_id: str, payload: Dict[str, Any]) -> None:
        self.state_store.create_job(job_id, payload)
        self.observer.emit(job_id, "commit", "prepare", payload)

    def vector_applied(self, job_id: str, payload: Dict[str, Any]) -> None:
        self.state_store.update_phase(job_id, "vector_applied", payload_patch=payload)
        self.observer.emit(job_id, "commit", "vector_applied", payload)

    def graph_applied(self, job_id: str, payload: Dict[str, Any]) -> None:
        self.state_store.update_phase(job_id, "graph_applied", payload_patch=payload)
        self.observer.emit(job_id, "commit", "graph_applied", payload)

    def confirm(self, job_id: str, payload: Dict[str, Any]) -> None:
        self.state_store.update_phase(job_id, "confirmed", payload_patch=payload)
        self.observer.emit(job_id, "commit", "confirmed", payload)

    def fail(self, job_id: str, payload: Dict[str, Any], error: str, phase: str) -> None:
        self.state_store.update_phase(job_id, phase, payload_patch=payload, error=error)
        self.observer.emit(job_id, "commit", "failed", {"phase": phase, "error": error})


class MemoryRuntime:
    def __init__(
        self,
        memory: Any,
        state_store: Any,
        prompt_registry: Any,
        observer: Any,
        graph_plugin_registry: Optional[GraphPluginRegistry] = None,
    ):
        self.memory = memory
        self.state_store = state_store
        self.prompt_registry = prompt_registry
        self.observer = observer
        self.commit = CommitCoordinator(state_store=state_store, observer=observer)
        self.graph_plugin_registry = graph_plugin_registry or DEFAULT_GRAPH_PLUGIN_REGISTRY
        plugins = resolve_graph_plugins(getattr(self.memory.config, "graph_pipeline", None), self.graph_plugin_registry)
        self.graph_entity_extractor = plugins["entity_extractor"]
        self.graph_relation_mapper = plugins["relation_mapper"]
        self.graph_entity_resolver = plugins["entity_resolver"]
        self.graph_mutation_planner = plugins["mutation_planner"]
        self.graph_writer = plugins["writer"]

    def _build_job_context(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> Dict[str, Any]:
        analysis_batch_id = metadata.get("analysis_batch_id") or metadata.get("analysisBatchId")
        return {
            "job_id": str(uuid.uuid4()),
            "analysis_batch_id": analysis_batch_id,
            "source_ref": metadata.get("source_ref"),
            "pipeline_version": "mem0_modular_v1",
            "user_id": filters.get("user_id", ""),
            "agent_id": filters.get("agent_id", ""),
            "run_id": filters.get("run_id", ""),
            "metadata": metadata,
        }

    def _graph_identity_keys(self, metadata: Dict[str, Any]) -> list[str]:
        keys = metadata.get("identity_keys")
        if isinstance(keys, list):
            return [str(item) for item in keys]
        return []

    def _resolve_stage_llm(self, stage_name: str, default_llm: Any) -> tuple[Any, str]:
        provider_routing = getattr(self.memory.config, "provider_routing", None)
        stage_mapping = {
            "semantic_fact_extraction": "semantic_fact_extraction",
            "graph_entity_extraction": "graph_entity_extraction",
            "graph_relation_calibration": "graph_relation_calibration",
            "summary_update_memory": "summary_update_memory",
        }
        config_key = stage_mapping.get(stage_name)
        if not config_key or provider_routing is None:
            return default_llm, getattr(self.memory.config.llm, "provider", "openai")

        route = getattr(provider_routing, config_key, None)
        if route is None:
            return default_llm, getattr(self.memory.config.llm, "provider", "openai")

        llm = self.memory._create_llm(route.provider, route.config)
        return llm, route.provider

    def _run_graph_pipeline(self, messages: list[dict], filters: Dict[str, Any], job_context: Dict[str, Any]) -> Dict[str, Any]:
        graph = self.memory.graph
        if graph is None:
            return {}
        if not _supports_modular_graph_pipeline(graph):
            self.observer.emit(
                job_context["job_id"],
                "GraphCompatibilityStage",
                "fallback",
                {"reason": "graph_backend_missing_modular_hooks"},
            )
            return self.memory._add_to_graph(messages, dict(filters))

        data = "\n".join([msg["content"] for msg in messages if "content" in msg and msg["role"] != "system"])
        graph_context = dict(job_context)
        graph_context["data"] = data
        graph_context["user_identity"] = ", ".join(
            f"{key}: {value}" for key, value in (("user_id", filters.get("user_id")), ("agent_id", filters.get("agent_id")), ("run_id", filters.get("run_id"))) if value
        )
        custom_prompt = getattr(self.memory.config.graph_store, "custom_prompt", None)
        graph_context["custom_prompt_line"] = f"4. {custom_prompt}" if custom_prompt else ""

        entity_llm, entity_provider = self._resolve_stage_llm("graph_entity_extraction", graph.llm)
        relation_llm, relation_provider = self._resolve_stage_llm("graph_relation_calibration", graph.llm)

        self.observer.emit(job_context["job_id"], "GraphEntityExtractionStage", "start", {"provider_name": entity_provider})
        entity_type_map = self.graph_entity_extractor.extract(graph, entity_llm, entity_provider, self.prompt_registry, graph_context)
        self.observer.emit(
            job_context["job_id"],
            "GraphEntityExtractionStage",
            "completed",
            {"provider_name": entity_provider, "entities": entity_type_map},
        )

        graph_context["entity_list"] = list(entity_type_map.keys())
        self.observer.emit(job_context["job_id"], "GraphRelationMappingStage", "start", {"provider_name": relation_provider})
        relations = self.graph_relation_mapper.map(graph, relation_llm, relation_provider, self.prompt_registry, graph_context)
        self.observer.emit(
            job_context["job_id"],
            "GraphRelationMappingStage",
            "completed",
            {"provider_name": relation_provider, "triplets": relations},
        )

        resolution_filters = dict(filters)
        resolution_filters["identity_keys"] = self._graph_identity_keys(job_context["metadata"])
        merge_candidates = self.graph_entity_resolver.resolve(
            graph,
            entity_type_map=entity_type_map,
            relations=relations,
            filters=resolution_filters,
            threshold=self.memory.config.graph_store.threshold,
        )
        self.observer.emit(
            job_context["job_id"],
            "GraphEntityResolutionStage",
            "completed",
            {"merge_candidates": merge_candidates},
        )

        plan = self.graph_mutation_planner.plan(
            graph,
            llm=relation_llm,
            llm_provider=relation_provider,
            prompt_registry=self.prompt_registry,
            entity_type_map=entity_type_map,
            relations=relations,
            filters=filters,
            context=graph_context,
        )
        plan["merge_candidates"] = merge_candidates
        self.observer.emit(
            job_context["job_id"],
            "GraphMutationPlanningStage",
            "completed",
            {"graph_mutation_plan": plan},
        )

        job_payload = self.state_store.get_job(job_context["job_id"]) or {"payload": {}}
        payload = job_payload.get("payload", {})
        payload["graph_plan"] = plan
        self.state_store.update_phase(job_context["job_id"], "vector_applied", payload_patch=payload)

        result = self.graph_writer.apply(graph, plan=plan, filters=filters, context=job_context)
        self.observer.emit(job_context["job_id"], "GraphWriterStage", "completed", result)
        return result

    def replay_graph(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self.state_store.get_job(job_id)
        if not job:
            return None
        graph_plan = job.get("payload", {}).get("graph_plan")
        filters = job.get("payload", {}).get("filters")
        job_context = job.get("payload", {}).get("job_context")
        if not graph_plan or not filters or not job_context or not self.memory.graph:
            return None
        result = self.graph_writer.apply(self.memory.graph, graph_plan, filters, job_context)
        self.commit.graph_applied(job_id, {"graph_result": result})
        self.commit.confirm(job_id, {"graph_result": result})
        return result

    def add(self, messages: list[dict], metadata: Dict[str, Any], filters: Dict[str, Any], infer: bool) -> Dict[str, Any]:
        self.prompt_registry.reload_if_needed()
        job_context = self._build_job_context(metadata, filters)
        metadata["job_id"] = job_context["job_id"]
        if job_context["analysis_batch_id"] and "analysis_batch_id" not in metadata:
            metadata["analysis_batch_id"] = job_context["analysis_batch_id"]
        self.commit.prepare(
            job_context["job_id"],
            {"filters": filters, "job_context": job_context, "messages": messages, "prompt_version": self.prompt_registry.snapshot().version},
        )

        try:
            self.observer.emit(job_context["job_id"], "VectorMutationPlanningStage", "start", {"infer": infer})
            vector_result = self.memory._add_to_vector_store(messages, metadata, filters, infer)
            self.observer.emit(
                job_context["job_id"],
                "VectorMutationPlanningStage",
                "completed",
                {"vector_write_payload": vector_result},
            )
            self.commit.vector_applied(job_context["job_id"], {"vector_result": vector_result})
        except Exception as exc:
            self.commit.fail(job_context["job_id"], {}, str(exc), "prepare")
            raise

        graph_result = None
        if self.memory.enable_graph:
            try:
                graph_result = self._run_graph_pipeline(messages, filters, job_context)
                self.commit.graph_applied(job_context["job_id"], {"graph_result": graph_result})
            except Exception as exc:
                self.commit.fail(job_context["job_id"], {}, str(exc), "vector_applied")
                raise

        self.commit.confirm(job_context["job_id"], {"graph_result": graph_result})
        if self.memory.enable_graph:
            return {"results": vector_result, "relations": graph_result}
        return {"results": vector_result}


class AsyncMemoryRuntime(MemoryRuntime):
    async def add(self, messages: list[dict], metadata: Dict[str, Any], filters: Dict[str, Any], infer: bool) -> Dict[str, Any]:
        self.prompt_registry.reload_if_needed()
        job_context = self._build_job_context(metadata, filters)
        metadata["job_id"] = job_context["job_id"]
        if job_context["analysis_batch_id"] and "analysis_batch_id" not in metadata:
            metadata["analysis_batch_id"] = job_context["analysis_batch_id"]

        self.commit.prepare(
            job_context["job_id"],
            {"filters": filters, "job_context": job_context, "messages": messages, "prompt_version": self.prompt_registry.snapshot().version},
        )

        try:
            self.observer.emit(job_context["job_id"], "VectorMutationPlanningStage", "start", {"infer": infer})
            vector_result = await self.memory._add_to_vector_store(messages, metadata, filters, infer)
            self.observer.emit(
                job_context["job_id"],
                "VectorMutationPlanningStage",
                "completed",
                {"vector_write_payload": vector_result},
            )
            self.commit.vector_applied(job_context["job_id"], {"vector_result": vector_result})
        except Exception as exc:
            self.commit.fail(job_context["job_id"], {}, str(exc), "prepare")
            raise

        graph_result = None
        if self.memory.enable_graph:
            try:
                graph_result = await asyncio.to_thread(self._run_graph_pipeline, messages, filters, job_context)
                self.commit.graph_applied(job_context["job_id"], {"graph_result": graph_result})
            except Exception as exc:
                self.commit.fail(job_context["job_id"], {}, str(exc), "vector_applied")
                raise

        self.commit.confirm(job_context["job_id"], {"graph_result": graph_result})
        if self.memory.enable_graph:
            return {"results": vector_result, "relations": graph_result}
        return {"results": vector_result}
