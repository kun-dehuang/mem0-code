from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional

from mem0.graphs.tools import (
    DELETE_MEMORY_STRUCT_TOOL_GRAPH,
    DELETE_MEMORY_TOOL_GRAPH,
    EXTRACT_ENTITIES_STRUCT_TOOL,
    EXTRACT_ENTITIES_TOOL,
    RELATIONS_STRUCT_TOOL,
    RELATIONS_TOOL,
)


def _structured_tools(llm_provider: str, regular_tool: Dict[str, Any], structured_tool: Dict[str, Any]) -> list[Dict[str, Any]]:
    if llm_provider in {"azure_openai_structured", "openai_structured"}:
        return [structured_tool]
    return [regular_tool]


class GraphEntityExtractor:
    def extract(self, graph: Any, llm: Any, llm_provider: str, prompt_registry: Any, context: Dict[str, Any]) -> Dict[str, str]:
        raise NotImplementedError


class GraphRelationMapper:
    def map(self, graph: Any, llm: Any, llm_provider: str, prompt_registry: Any, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError


class GraphEntityResolver:
    def resolve(self, graph: Any, entity_type_map: Dict[str, str], relations: List[Dict[str, Any]], filters: Dict[str, Any], threshold: float) -> List[Dict[str, Any]]:
        raise NotImplementedError


class GraphMutationPlanner:
    def plan(
        self,
        graph: Any,
        llm: Any,
        llm_provider: str,
        prompt_registry: Any,
        entity_type_map: Dict[str, str],
        relations: List[Dict[str, Any]],
        filters: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        raise NotImplementedError


class GraphWriter:
    def apply(self, graph: Any, plan: Dict[str, Any], filters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class DefaultGraphEntityExtractor(GraphEntityExtractor):
    def extract(self, graph: Any, llm: Any, llm_provider: str, prompt_registry: Any, context: Dict[str, Any]) -> Dict[str, str]:
        tools = _structured_tools(llm_provider, EXTRACT_ENTITIES_TOOL, EXTRACT_ENTITIES_STRUCT_TOOL)
        response = llm.generate_response(
            messages=[
                {"role": "system", "content": prompt_registry.render("graph.entity_extraction.system", context)},
                {"role": "user", "content": context["data"]},
            ],
            tools=tools,
        )

        entity_type_map: Dict[str, str] = {}
        for tool_call in response.get("tool_calls", []):
            if tool_call.get("name") != "extract_entities":
                continue
            for item in tool_call.get("arguments", {}).get("entities", []):
                if "entity" in item and "entity_type" in item:
                    entity_type_map[item["entity"]] = item["entity_type"]

        return {k.lower().replace(" ", "_"): v.lower().replace(" ", "_") for k, v in entity_type_map.items()}


class DefaultGraphRelationMapper(GraphRelationMapper):
    def map(self, graph: Any, llm: Any, llm_provider: str, prompt_registry: Any, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        tools = _structured_tools(llm_provider, RELATIONS_TOOL, RELATIONS_STRUCT_TOOL)
        messages = [{"role": "system", "content": prompt_registry.render("graph.relation.system", context)}]

        if context.get("custom_prompt_line"):
            messages.append({"role": "user", "content": context["data"]})
        else:
            messages.append(
                {
                    "role": "user",
                    "content": prompt_registry.render("graph.relation.user_with_entities", context),
                }
            )

        response = llm.generate_response(messages=messages, tools=tools)
        entities = []
        if response.get("tool_calls"):
            entities = response["tool_calls"][0].get("arguments", {}).get("entities", [])
        return graph._remove_spaces_from_entities(entities)


class SemanticSimilarityGraphEntityResolver(GraphEntityResolver):
    def resolve(self, graph: Any, entity_type_map: Dict[str, str], relations: List[Dict[str, Any]], filters: Dict[str, Any], threshold: float) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for relation in relations:
            source_embedding = graph.embedding_model.embed(relation["source"])
            destination_embedding = graph.embedding_model.embed(relation["destination"])
            candidates.append(
                {
                    "source": relation["source"],
                    "relationship": relation["relationship"],
                    "destination": relation["destination"],
                    "source_matches": graph._search_source_node(source_embedding, filters, threshold=threshold),
                    "destination_matches": graph._search_destination_node(destination_embedding, filters, threshold=threshold),
                }
            )
        return candidates


class StrictIdentityGraphEntityResolver(GraphEntityResolver):
    def resolve(self, graph: Any, entity_type_map: Dict[str, str], relations: List[Dict[str, Any]], filters: Dict[str, Any], threshold: float) -> List[Dict[str, Any]]:
        identity_keys = set((filters.get("identity_keys") or []))
        results: List[Dict[str, Any]] = []
        for relation in relations:
            strict_match = relation["source"] in identity_keys or relation["destination"] in identity_keys
            results.append(
                {
                    "source": relation["source"],
                    "relationship": relation["relationship"],
                    "destination": relation["destination"],
                    "strict_identity_match": strict_match,
                }
            )
        return results


class DefaultGraphMutationPlanner(GraphMutationPlanner):
    def plan(
        self,
        graph: Any,
        llm: Any,
        llm_provider: str,
        prompt_registry: Any,
        entity_type_map: Dict[str, str],
        relations: List[Dict[str, Any]],
        filters: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        search_output = graph._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)
        planner_context = dict(context)
        planner_context["existing_memories_string"] = graph.__class__.__module__.endswith("graph_memory") and "\n".join(
            f"{item['source']} -- {item['relationship']} -- {item['destination']}" for item in search_output
        ) or planner_context.get("existing_memories_string", "")

        tools = _structured_tools(llm_provider, DELETE_MEMORY_TOOL_GRAPH, DELETE_MEMORY_STRUCT_TOOL_GRAPH)
        delete_response = llm.generate_response(
            messages=[
                {"role": "system", "content": prompt_registry.render("graph.delete.system", planner_context)},
                {"role": "user", "content": prompt_registry.render("graph.delete.user", planner_context)},
            ],
            tools=tools,
        )
        to_delete = []
        for item in delete_response.get("tool_calls", []):
            if item.get("name") == "delete_graph_memory":
                to_delete.append(item.get("arguments"))
        to_delete = graph._remove_spaces_from_entities(to_delete)
        return {
            "entity_type_map": entity_type_map,
            "relations_to_add": relations,
            "relations_to_delete": to_delete,
            "search_output": search_output,
            "delete_raw_output": delete_response,
        }


class DefaultGraphWriter(GraphWriter):
    def apply(self, graph: Any, plan: Dict[str, Any], filters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        deleted = graph._delete_entities(plan.get("relations_to_delete", []), filters)
        added = graph._add_entities(plan.get("relations_to_add", []), filters, plan.get("entity_type_map", {}))
        _apply_neo4j_provenance(graph, plan.get("relations_to_add", []), filters, context)
        return {
            "deleted_entities": deleted,
            "added_entities": added,
        }


class GraphPluginRegistry:
    def __init__(self):
        self.entity_extractors: Dict[str, type[GraphEntityExtractor]] = {"default": DefaultGraphEntityExtractor}
        self.relation_mappers: Dict[str, type[GraphRelationMapper]] = {"default": DefaultGraphRelationMapper}
        self.entity_resolvers: Dict[str, type[GraphEntityResolver]] = {
            "semantic_similarity": SemanticSimilarityGraphEntityResolver,
            "strict_identity_match": StrictIdentityGraphEntityResolver,
        }
        self.mutation_planners: Dict[str, type[GraphMutationPlanner]] = {"default": DefaultGraphMutationPlanner}
        self.writers: Dict[str, type[GraphWriter]] = {"default": DefaultGraphWriter}

    def register_entity_extractor(self, name: str, plugin_cls: type[GraphEntityExtractor]) -> None:
        self.entity_extractors[name] = plugin_cls

    def register_relation_mapper(self, name: str, plugin_cls: type[GraphRelationMapper]) -> None:
        self.relation_mappers[name] = plugin_cls

    def register_entity_resolver(self, name: str, plugin_cls: type[GraphEntityResolver]) -> None:
        self.entity_resolvers[name] = plugin_cls

    def register_mutation_planner(self, name: str, plugin_cls: type[GraphMutationPlanner]) -> None:
        self.mutation_planners[name] = plugin_cls

    def register_writer(self, name: str, plugin_cls: type[GraphWriter]) -> None:
        self.writers[name] = plugin_cls

    def resolve_entity_extractor(self, plugin: Any) -> GraphEntityExtractor:
        return self._resolve(plugin, self.entity_extractors, GraphEntityExtractor)

    def resolve_relation_mapper(self, plugin: Any) -> GraphRelationMapper:
        return self._resolve(plugin, self.relation_mappers, GraphRelationMapper)

    def resolve_entity_resolver(self, plugin: Any) -> GraphEntityResolver:
        return self._resolve(plugin, self.entity_resolvers, GraphEntityResolver)

    def resolve_mutation_planner(self, plugin: Any) -> GraphMutationPlanner:
        return self._resolve(plugin, self.mutation_planners, GraphMutationPlanner)

    def resolve_writer(self, plugin: Any) -> GraphWriter:
        return self._resolve(plugin, self.writers, GraphWriter)

    def _resolve(self, plugin: Any, registry: Dict[str, type[Any]], expected_type: type[Any]) -> Any:
        if isinstance(plugin, expected_type):
            return plugin

        if plugin is None:
            plugin = "default"

        if isinstance(plugin, str):
            plugin_cls = registry.get(plugin)
            if plugin_cls is None:
                plugin_cls = self._import_plugin(plugin)
            instance = plugin_cls()
        elif isinstance(plugin, type):
            instance = plugin()
        else:
            raise TypeError(f"Unsupported plugin specification: {plugin!r}")

        if not isinstance(instance, expected_type):
            raise TypeError(f"Plugin {plugin!r} must implement {expected_type.__name__}")
        return instance

    @staticmethod
    def _import_plugin(import_path: str) -> type[Any]:
        module_path, separator, attr_name = import_path.replace(":", ".").rpartition(".")
        if not separator:
            raise KeyError(f"Unknown graph plugin: {import_path}")
        module = importlib.import_module(module_path)
        plugin_cls = getattr(module, attr_name)
        if not isinstance(plugin_cls, type):
            raise TypeError(f"Imported graph plugin {import_path} is not a class")
        return plugin_cls


DEFAULT_GRAPH_PLUGIN_REGISTRY = GraphPluginRegistry()


def resolve_graph_plugins(graph_pipeline_config: Any, registry: Optional[GraphPluginRegistry] = None) -> Dict[str, Any]:
    graph_pipeline_config = graph_pipeline_config or object()
    registry = registry or DEFAULT_GRAPH_PLUGIN_REGISTRY
    return {
        "entity_extractor": registry.resolve_entity_extractor(getattr(graph_pipeline_config, "entity_extractor", "default")),
        "relation_mapper": registry.resolve_relation_mapper(getattr(graph_pipeline_config, "relation_mapper", "default")),
        "entity_resolver": registry.resolve_entity_resolver(
            getattr(graph_pipeline_config, "entity_resolver", "semantic_similarity")
        ),
        "mutation_planner": registry.resolve_mutation_planner(getattr(graph_pipeline_config, "mutation_planner", "default")),
        "writer": registry.resolve_writer(getattr(graph_pipeline_config, "writer", "default")),
    }


def _apply_neo4j_provenance(graph: Any, relations: List[Dict[str, Any]], filters: Dict[str, Any], context: Dict[str, Any]) -> None:
    graph_client = getattr(graph, "graph", None)
    if not graph_client or not hasattr(graph_client, "query"):
        return

    job_id = context.get("job_id")
    if not job_id:
        return

    source_ref = context.get("source_ref")
    analysis_batch_id = context.get("analysis_batch_id")
    pipeline_version = context.get("pipeline_version")
    user_id = filters.get("user_id")
    agent_id = filters.get("agent_id")
    run_id = filters.get("run_id")

    graph_client.query(
        """
        MERGE (job:IngestionJob {job_id: $job_id})
        SET job.analysis_batch_id = $analysis_batch_id,
            job.source_ref = $source_ref,
            job.pipeline_version = $pipeline_version,
            job.user_id = $user_id,
            job.agent_id = $agent_id,
            job.run_id = $run_id
        """,
        params={
            "job_id": job_id,
            "analysis_batch_id": analysis_batch_id,
            "source_ref": source_ref,
            "pipeline_version": pipeline_version,
            "user_id": user_id,
            "agent_id": agent_id,
            "run_id": run_id,
        },
    )

    for relation in relations:
        relationship = relation["relationship"]
        graph_client.query(
            f"""
            MATCH (src {{name: $source_name, user_id: $user_id}})
            MATCH (dst {{name: $destination_name, user_id: $user_id}})
            MATCH (src)-[rel:{relationship}]->(dst)
            MERGE (job:IngestionJob {{job_id: $job_id}})
            SET src.job_id = $job_id,
                src.analysis_batch_id = $analysis_batch_id,
                src.source_ref = $source_ref,
                src.pipeline_version = $pipeline_version,
                dst.job_id = $job_id,
                dst.analysis_batch_id = $analysis_batch_id,
                dst.source_ref = $source_ref,
                dst.pipeline_version = $pipeline_version,
                rel.job_id = $job_id,
                rel.analysis_batch_id = $analysis_batch_id,
                rel.source_ref = $source_ref,
                rel.pipeline_version = $pipeline_version
            MERGE (src)-[:GENERATED_IN]->(job)
            MERGE (dst)-[:GENERATED_IN]->(job)
            """,
            params={
                "job_id": job_id,
                "analysis_batch_id": analysis_batch_id,
                "source_ref": source_ref,
                "pipeline_version": pipeline_version,
                "source_name": relation["source"],
                "destination_name": relation["destination"],
                "user_id": user_id,
            },
        )
