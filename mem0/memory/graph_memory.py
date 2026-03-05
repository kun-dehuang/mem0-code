import logging

from mem0.memory.utils import format_entities, sanitize_relationship_for_cypher

try:
    from langchain_neo4j import Neo4jGraph
except ImportError:
    raise ImportError("langchain_neo4j is not installed. Please install it using pip install langchain-neo4j")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank_bm25 is not installed. Please install it using pip install rank-bm25")

from mem0.graphs.tools import (
    DELETE_MEMORY_STRUCT_TOOL_GRAPH,
    DELETE_MEMORY_TOOL_GRAPH,
    EXTRACT_ENTITIES_STRUCT_TOOL,
    EXTRACT_ENTITIES_TOOL,
    RELATIONS_STRUCT_TOOL,
    RELATIONS_TOOL,
)
from mem0.graphs.utils import EXTRACT_RELATIONS_PROMPT, get_delete_messages
from mem0.utils.factory import EmbedderFactory, LlmFactory
# 获取与 memory-select 相同的 logger
logger = logging.getLogger("memory_debug")


class MemoryGraph:
    def __init__(self, config):
        self.config = config
        self.graph = Neo4jGraph(
            url=self.config.graph_store.config.url,
            username=self.config.graph_store.config.username,
            password=self.config.graph_store.config.password,
            database=self.config.graph_store.config.database,
            refresh_schema=False,
            driver_config={"notifications_min_severity": "OFF"},
        )
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider, self.config.embedder.config, self.config.vector_store.config
        )
        self.node_label = ":`__Entity__`" if self.config.graph_store.config.base_label else ""

        if self.config.graph_store.config.base_label:
            # Safely add user_id index
            try:
                self.graph.query(f"CREATE INDEX entity_single IF NOT EXISTS FOR (n {self.node_label}) ON (n.user_id)")
            except Exception:
                pass
            try:  # Safely try to add composite index (Enterprise only)
                self.graph.query(
                    f"CREATE INDEX entity_composite IF NOT EXISTS FOR (n {self.node_label}) ON (n.name, n.user_id)"
                )
            except Exception:
                pass

        # Default to openai if no specific provider is configured
        self.llm_provider = "openai"
        if self.config.llm and self.config.llm.provider:
            self.llm_provider = self.config.llm.provider
        if self.config.graph_store and self.config.graph_store.llm and self.config.graph_store.llm.provider:
            self.llm_provider = self.config.graph_store.llm.provider

        # Get LLM config with proper null checks
        llm_config = None
        if self.config.graph_store and self.config.graph_store.llm and hasattr(self.config.graph_store.llm, "config"):
            llm_config = self.config.graph_store.llm.config
        elif hasattr(self.config.llm, "config"):
            llm_config = self.config.llm.config
        self.llm = LlmFactory.create(self.llm_provider, llm_config)
        self.user_id = None
        # Use threshold from graph_store config, default to 0.7 for backward compatibility
        self.threshold = self.config.graph_store.threshold if hasattr(self.config.graph_store, 'threshold') else 0.7

    def add(self, data, filters):
        """
        Adds data to the graph.

        Args:
            data (str): The data to add to the graph.
            filters (dict): A dictionary containing filters to be applied during the addition.
        """

        # 获取与 memory-select 相同的 logger
        logger = logging.getLogger("memory_debug")

        # 然后就可以写入日志了
        logger.info("[MEM0 INTERNAL] add")

        entity_type_map = self._retrieve_nodes_from_data(data, filters)
        to_be_added = self._establish_nodes_relations_from_data(data, filters, entity_type_map)
        search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)
        to_be_deleted = self._get_delete_entities_from_search_output(search_output, data, filters)

        # TODO: Batch queries with APOC plugin
        # TODO: Add more filter support
        deleted_entities = self._delete_entities(to_be_deleted, filters)
        added_entities = self._add_entities(to_be_added, filters, entity_type_map)

        return {"deleted_entities": deleted_entities, "added_entities": added_entities}

    def search(self, query, filters, limit=100):
        """
        Search for memories and related graph data.

        Args:
            query (str): Query to search for.
            filters (dict): A dictionary containing filters to be applied during the search.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.

        Returns:
            dict: A dictionary containing:
                - "contexts": List of search results from the base data store.
                - "entities": List of related graph data based on the query.
        """
        # 然后就可以写入日志了
        logger.info("=" * 80)
        logger.info("[GRAPH SEARCH] Starting graph search")
        logger.info(f"[GRAPH SEARCH] query: '{query}'")
        logger.info(f"[GRAPH SEARCH] filters: {filters}")
        logger.info(f"[GRAPH SEARCH] limit: {limit}")

        entity_type_map = self._retrieve_nodes_from_data(query, filters)
        logger.info(f"[GRAPH SEARCH] Extracted entity_type_map: {entity_type_map}")
        logger.info(f"[GRAPH SEARCH] entity_type_map keys: {list(entity_type_map.keys())}")

        node_list = list(entity_type_map.keys())
        if not node_list:
            logger.warning("[GRAPH SEARCH] No entities extracted from query! Search will return empty results.")
            logger.warning("[GRAPH SEARCH] This is likely because LLM did not return any tool_calls in _retrieve_nodes_from_data")
            return []

        search_output = self._search_graph_db(node_list=node_list, filters=filters)
        logger.info(f"[GRAPH SEARCH] _search_graph_db returned {len(search_output)} results")

        if not search_output:
            return []

        search_outputs_sequence = [
            [item["source"], item["relationship"], item["destination"]] for item in search_output
        ]
        bm25 = BM25Okapi(search_outputs_sequence)

        tokenized_query = query.split(" ")
        reranked_results = bm25.get_top_n(tokenized_query, search_outputs_sequence, n=5)
        logger.info(f"Reranked results: {reranked_results}")

        search_results = []
        for item in reranked_results:
            search_results.append({"source": item[0], "relationship": item[1], "destination": item[2]})

        logger.info(f"Returned {len(search_results)} search results")

        return search_results

    def delete_all(self, filters):
        # Build node properties for filtering
        node_props = ["user_id: $user_id"]
        if filters.get("agent_id"):
            node_props.append("agent_id: $agent_id")
        if filters.get("run_id"):
            node_props.append("run_id: $run_id")
        node_props_str = ", ".join(node_props)

        cypher = f"""
        MATCH (n {self.node_label} {{{node_props_str}}})
        DETACH DELETE n
        """
        params = {"user_id": filters["user_id"]}
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]
        self.graph.query(cypher, params=params)

    def get_all(self, filters, limit=100):
        """
        Retrieves all nodes and relationships from the graph database based on optional filtering criteria.
         Args:
            filters (dict): A dictionary containing filters to be applied during the retrieval.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.
        Returns:
            list: A list of dictionaries, each containing:
                - 'contexts': The base data store response for each memory.
                - 'entities': A list of strings representing the nodes and relationships
        """
        params = {"user_id": filters["user_id"], "limit": limit}

        # Build node properties based on filters
        node_props = ["user_id: $user_id"]
        if filters.get("agent_id"):
            node_props.append("agent_id: $agent_id")
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            node_props.append("run_id: $run_id")
            params["run_id"] = filters["run_id"]
        node_props_str = ", ".join(node_props)

        query = f"""
        MATCH (n {self.node_label} {{{node_props_str}}})-[r]->(m {self.node_label} {{{node_props_str}}})
        RETURN n.name AS source, type(r) AS relationship, m.name AS target
        LIMIT $limit
        """
        results = self.graph.query(query, params=params)

        final_results = []
        for result in results:
            final_results.append(
                {
                    "source": result["source"],
                    "relationship": result["relationship"],
                    "target": result["target"],
                }
            )

        logger.info(f"Retrieved {len(final_results)} relationships")

        return final_results

    def _retrieve_nodes_from_data(self, data, filters):
        """Extracts all the entities mentioned in the query."""
        logger.info(f"[ENTITY EXTRACTION] Input data: '{data[:200]}...'")  # Log first 200 chars
        logger.info(f"[ENTITY EXTRACTION] Filters: {filters}")
        logger.info(f"[ENTITY EXTRACTION] LLM provider: {self.llm_provider}")

        _tools = [EXTRACT_ENTITIES_TOOL]
        nodes_prompt = f'''
### Task
Extract entities from the text and classify them into EXACTLY ONE of the 4 predefined categories below.

### Entity Type Categories (MUST use one of these)
1. **person** - Individual people, social roles, or the user themselves
2. **place** - Geographic locations, venues, buildings, or points of interest
3. **object** - All other entities including food, organizations, events, apps, products, projects
4. **category** - Abstract concepts, interests, topics, or fields of study

### Special Rules
1. Self-Reference (I, me, my, myself) → entity: "{filters['user_id']}", type: "person"
2. User mentioned by name → entity: "{filters['user_id']}", type: "person"
3. If input is a QUESTION, DO NOT answer it. Only extract entities mentioned.
4. Each entity MUST have ONE of the 4 types above. NO OTHER types allowed.
5. DO NOT extract verbs or actions as entities (e.g., "walking", "running", "discussing").
   Instead, extract related objects or places (e.g., from "walk in the park", extract "park" as place).
6. DO NOT extract time expressions, dates, or temporal words as entities.
   Examples to EXCLUDE:
   - Relative time: tomorrow, yesterday, next week, last month, soon, later
   - Absolute time: January 2024, 3pm, Monday, 2025年
   - Time-related words: appointment, schedule, deadline, time, date

### Examples

Input: "My boss and I discussed Project X at a cafe in Barcelona."
Output:
- {filters['user_id']} | person
- boss | person
- project_x | object
- cafe | place
- barcelona | place

Input: "I love photography, sunset views, and eating sushi."
Output:
- {filters['user_id']} | person
- photography | category
- sunset | category
- sushi | object

Input: "I posted a photo on Instagram about the company retreat."
Output:
- {filters['user_id']} | person
- photo | object
- instagram | object
- company_retreat | object

Input: "What is my doctor's appointment at the hospital schedule?"
Output:
- {filters['user_id']} | person
- doctor | person
- hospital | object

Input: "I had coffee at Starbucks near my office."
Output:
- {filters['user_id']} | person
- coffee | object
- starbucks | object
- office | place

Now extract entities from the text above, using ONLY the 4 allowed types.
        '''
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [EXTRACT_ENTITIES_STRUCT_TOOL]
        search_results = self.llm.generate_response(
            messages=[
                {
                    "role": "system",
                    "content": nodes_prompt,
                },
                {"role": "user", "content": data},
            ],
            tools=_tools,
        )

        # Debug log: log search_results structure
        logger.info(f"[ENTITY EXTRACTION] search_results type: {type(search_results)}")
        logger.info(f"[ENTITY EXTRACTION] search_results keys: {search_results.keys() if isinstance(search_results, dict) else 'N/A'}")
        logger.info(f"[ENTITY EXTRACTION] search_results content: {search_results}")

        entity_type_map = {}

        try:
            tool_calls = search_results.get("tool_calls", [])
            logger.info(f"[ENTITY EXTRACTION] tool_calls count: {len(tool_calls)}")

            if not tool_calls:
                logger.warning("[ENTITY EXTRACTION] No tool_calls in LLM response! This is the problem!")
                logger.warning("[ENTITY EXTRACTION] LLM may have returned text directly instead of using tools")
                # Check if there's a 'content' field
                if isinstance(search_results, dict) and "content" in search_results:
                    content = search_results['content']
                    logger.warning(f"[ENTITY EXTRACTION] LLM returned content instead of tool_calls: {content[:500]}")
                    # Try to parse JSON from content (Gemini sometimes returns JSON in content instead of tool_calls)
                    import json
                    import re
                    parsed = False

                    # Method 1: Try JSON array in markdown code blocks
                    try:
                        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                            entities = json.loads(json_str)
                            logger.info(f"[ENTITY EXTRACTION] Successfully parsed {len(entities)} entities from JSON markdown")
                            for item in entities:
                                entity_type_map[item["entity"]] = item["entity_type"]
                            parsed = True
                    except (json.JSONDecodeError, KeyError, Exception) as e:
                        logger.debug(f"[ENTITY EXTRACTION] Method 1 failed: {e}")

                    # Method 2: Try plain JSON array
                    if not parsed:
                        try:
                            json_match = re.search(r'\[\s*\{.*?\}\s*\]', content, re.DOTALL)
                            if json_match:
                                json_str = json_match.group(0)
                                entities = json.loads(json_str)
                                logger.info(f"[ENTITY EXTRACTION] Successfully parsed {len(entities)} entities from plain JSON")
                                for item in entities:
                                    entity_type_map[item["entity"]] = item["entity_type"]
                                parsed = True
                        except (json.JSONDecodeError, KeyError, Exception) as e:
                            logger.debug(f"[ENTITY EXTRACTION] Method 2 failed: {e}")

                    # Method 3: Parse bullet point format "- entity | type"
                    if not parsed:
                        try:
                            lines = content.strip().split('\n')
                            for line in lines:
                                line = line.strip()
                                # Match "- entity | type" or "* entity | type"
                                match = re.match(r'^[\-\*]\s*(.+?)\s*\|\s*(.+)$', line)
                                if match:
                                    entity = match.group(1).strip()
                                    entity_type = match.group(2).strip()
                                    entity_type_map[entity] = entity_type
                            if entity_type_map:
                                logger.info(f"[ENTITY EXTRACTION] Successfully parsed {len(entity_type_map)} entities from bullet format")
                                parsed = True
                        except Exception as e:
                            logger.debug(f"[ENTITY EXTRACTION] Method 3 failed: {e}")

                    if not parsed:
                        logger.warning(f"[ENTITY EXTRACTION] All parsing methods failed")

            for tool_call in tool_calls:
                logger.info(f"[ENTITY EXTRACTION] tool_call: {tool_call}")
                if tool_call.get("name") != "extract_entities":
                    logger.info(f"[ENTITY EXTRACTION] Skipping tool: {tool_call.get('name')}")
                    continue
                args = tool_call.get("arguments", {})
                entities = args.get("entities", [])
                logger.info(f"[ENTITY EXTRACTION] extract_entities arguments: {args}")
                logger.info(f"[ENTITY EXTRACTION] entities count: {len(entities)}")
                for item in entities:
                    entity_type_map[item["entity"]] = item["entity_type"]
        except Exception as e:
            logger.exception(
                f"Error in search tool: {e}, llm_provider={self.llm_provider}, search_results={search_results}"
            )

        entity_type_map = {k.lower().replace(" ", "_"): v.lower().replace(" ", "_") for k, v in entity_type_map.items()}
        logger.info(f"[ENTITY EXTRACTION] Final entity_type_map: {entity_type_map}")
        logger.info(f"[ENTITY EXTRACTION] entity_type_map size: {len(entity_type_map)}")
        if not entity_type_map:
            logger.error("[ENTITY EXTRACTION] entity_type_map is EMPTY! No entities were extracted!")
        return entity_type_map

    def _establish_nodes_relations_from_data(self, data, filters, entity_type_map):
        """Establish relations among the extracted nodes."""

        # Compose user identification string for prompt
        user_identity = f"user_id: {filters['user_id']}"
        if filters.get("agent_id"):
            user_identity += f", agent_id: {filters['agent_id']}"
        if filters.get("run_id"):
            user_identity += f", run_id: {filters['run_id']}"

        if self.config.graph_store.custom_prompt:
            system_content = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_identity)
            # Add the custom prompt line if configured
            system_content = system_content.replace("CUSTOM_PROMPT", f"4. {self.config.graph_store.custom_prompt}")
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": data},
            ]
        else:
            system_content = EXTRACT_RELATIONS_PROMPT.replace("USER_ID", user_identity)
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"List of entities: {list(entity_type_map.keys())}. \n\nText: {data}"},
            ]

        _tools = [RELATIONS_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [RELATIONS_STRUCT_TOOL]

        extracted_entities = self.llm.generate_response(
            messages=messages,
            tools=_tools,
        )

        entities = []
        if extracted_entities.get("tool_calls"):
            entities = extracted_entities["tool_calls"][0].get("arguments", {}).get("entities", [])

        entities = self._remove_spaces_from_entities(entities)
        logger.debug(f"Extracted entities: {entities}")
        return entities

    def _search_graph_db(self, node_list, filters, limit=100):
        """Search similar nodes among and their respective incoming and outgoing relations."""
        result_relations = []

        # Build node properties for filtering
        node_props = ["user_id: $user_id"]
        if filters.get("agent_id"):
            node_props.append("agent_id: $agent_id")
        if filters.get("run_id"):
            node_props.append("run_id: $run_id")
        node_props_str = ", ".join(node_props)

        for node in node_list:
            n_embedding = self.embedding_model.embed(node)

            cypher_query = f"""
            MATCH (n {self.node_label} {{{node_props_str}}})
            WHERE n.embedding IS NOT NULL
            WITH n, round(2 * vector.similarity.cosine(n.embedding, $n_embedding) - 1, 4) AS similarity // denormalize for backward compatibility
            WHERE similarity >= $threshold
            CALL {{
                WITH n
                MATCH (n)-[r]->(m {self.node_label} {{{node_props_str}}})
                RETURN n.name AS source, elementId(n) AS source_id, type(r) AS relationship, elementId(r) AS relation_id, m.name AS destination, elementId(m) AS destination_id
                UNION
                WITH n  
                MATCH (n)<-[r]-(m {self.node_label} {{{node_props_str}}})
                RETURN m.name AS source, elementId(m) AS source_id, type(r) AS relationship, elementId(r) AS relation_id, n.name AS destination, elementId(n) AS destination_id
            }}
            WITH distinct source, source_id, relationship, relation_id, destination, destination_id, similarity
            RETURN source, source_id, relationship, relation_id, destination, destination_id, similarity
            ORDER BY similarity DESC
            LIMIT $limit
            """

            logger.info(f"Search Graph Node: {node}")

            params = {
                "n_embedding": n_embedding,
                "threshold": self.threshold,
                "user_id": filters["user_id"],
                "limit": limit,
            }
            if filters.get("agent_id"):
                params["agent_id"] = filters["agent_id"]
            if filters.get("run_id"):
                params["run_id"] = filters["run_id"]

            ans = self.graph.query(cypher_query, params=params)
            result_relations.extend(ans)
            logger.info(f"Node: {node}, relations: {ans}")

        return result_relations

    def _get_delete_entities_from_search_output(self, search_output, data, filters):
        """Get the entities to be deleted from the search output."""
        search_output_string = format_entities(search_output)

        # Compose user identification string for prompt
        user_identity = f"user_id: {filters['user_id']}"
        if filters.get("agent_id"):
            user_identity += f", agent_id: {filters['agent_id']}"
        if filters.get("run_id"):
            user_identity += f", run_id: {filters['run_id']}"

        system_prompt, user_prompt = get_delete_messages(search_output_string, data, user_identity)

        _tools = [DELETE_MEMORY_TOOL_GRAPH]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [
                DELETE_MEMORY_STRUCT_TOOL_GRAPH,
            ]

        memory_updates = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=_tools,
        )

        to_be_deleted = []
        for item in memory_updates.get("tool_calls", []):
            if item.get("name") == "delete_graph_memory":
                to_be_deleted.append(item.get("arguments"))
        # Clean entities formatting
        to_be_deleted = self._remove_spaces_from_entities(to_be_deleted)
        logger.debug(f"Deleted relationships: {to_be_deleted}")
        return to_be_deleted

    def _delete_entities(self, to_be_deleted, filters):
        """Delete the entities from the graph."""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        run_id = filters.get("run_id", None)
        results = []

        for item in to_be_deleted:
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # Build the agent filter for the query

            params = {
                "source_name": source,
                "dest_name": destination,
                "user_id": user_id,
            }

            if agent_id:
                params["agent_id"] = agent_id
            if run_id:
                params["run_id"] = run_id

            # Build node properties for filtering
            source_props = ["name: $source_name", "user_id: $user_id"]
            dest_props = ["name: $dest_name", "user_id: $user_id"]
            if agent_id:
                source_props.append("agent_id: $agent_id")
                dest_props.append("agent_id: $agent_id")
            if run_id:
                source_props.append("run_id: $run_id")
                dest_props.append("run_id: $run_id")
            source_props_str = ", ".join(source_props)
            dest_props_str = ", ".join(dest_props)

            # Delete the specific relationship between nodes
            cypher = f"""
            MATCH (n {self.node_label} {{{source_props_str}}})
            -[r:{relationship}]->
            (m {self.node_label} {{{dest_props_str}}})
            
            DELETE r
            RETURN 
                n.name AS source,
                m.name AS target,
                type(r) AS relationship
            """

            result = self.graph.query(cypher, params=params)
            results.append(result)

        return results

    def _add_entities(self, to_be_added, filters, entity_type_map):
        """Add the new entities to the graph. Merge the nodes if they already exist."""
        user_id = filters["user_id"]
        agent_id = filters.get("agent_id", None)
        run_id = filters.get("run_id", None)
        results = []
        for item in to_be_added:
            # entities
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # types
            source_type = entity_type_map.get(source, "__User__")
            source_label = self.node_label if self.node_label else f":`{source_type}`"
            source_extra_set = f", source:`{source_type}`" if self.node_label else ""
            destination_type = entity_type_map.get(destination, "__User__")
            destination_label = self.node_label if self.node_label else f":`{destination_type}`"
            destination_extra_set = f", destination:`{destination_type}`" if self.node_label else ""

            # embeddings
            source_embedding = self.embedding_model.embed(source)
            dest_embedding = self.embedding_model.embed(destination)

            # search for the nodes with the closest embeddings
            source_node_search_result = self._search_source_node(source_embedding, filters, threshold=self.threshold)
            destination_node_search_result = self._search_destination_node(dest_embedding, filters, threshold=self.threshold)

            # TODO: Create a cypher query and common params for all the cases
            if not destination_node_search_result and source_node_search_result:
                # Build destination MERGE properties
                merge_props = ["name: $destination_name", "user_id: $user_id"]
                if agent_id:
                    merge_props.append("agent_id: $agent_id")
                if run_id:
                    merge_props.append("run_id: $run_id")
                merge_props_str = ", ".join(merge_props)

                cypher = f"""
                MATCH (source)
                WHERE elementId(source) = $source_id
                SET source.mentions = coalesce(source.mentions, 0) + 1
                WITH source
                MERGE (destination {destination_label} {{{merge_props_str}}})
                ON CREATE SET
                    destination.created = timestamp(),
                    destination.mentions = 1
                    {destination_extra_set}
                ON MATCH SET
                    destination.mentions = coalesce(destination.mentions, 0) + 1
                WITH source, destination
                CALL db.create.setNodeVectorProperty(destination, 'embedding', $destination_embedding)
                WITH source, destination
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET 
                    r.created = timestamp(),
                    r.mentions = 1
                ON MATCH SET
                    r.mentions = coalesce(r.mentions, 0) + 1
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

                params = {
                    "source_id": source_node_search_result[0]["elementId(source_candidate)"],
                    "destination_name": destination,
                    "destination_embedding": dest_embedding,
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id

            elif destination_node_search_result and not source_node_search_result:
                # Build source MERGE properties
                merge_props = ["name: $source_name", "user_id: $user_id"]
                if agent_id:
                    merge_props.append("agent_id: $agent_id")
                if run_id:
                    merge_props.append("run_id: $run_id")
                merge_props_str = ", ".join(merge_props)

                cypher = f"""
                MATCH (destination)
                WHERE elementId(destination) = $destination_id
                SET destination.mentions = coalesce(destination.mentions, 0) + 1
                WITH destination
                MERGE (source {source_label} {{{merge_props_str}}})
                ON CREATE SET
                    source.created = timestamp(),
                    source.mentions = 1
                    {source_extra_set}
                ON MATCH SET
                    source.mentions = coalesce(source.mentions, 0) + 1
                WITH source, destination
                CALL db.create.setNodeVectorProperty(source, 'embedding', $source_embedding)
                WITH source, destination
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET 
                    r.created = timestamp(),
                    r.mentions = 1
                ON MATCH SET
                    r.mentions = coalesce(r.mentions, 0) + 1
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

                params = {
                    "destination_id": destination_node_search_result[0]["elementId(destination_candidate)"],
                    "source_name": source,
                    "source_embedding": source_embedding,
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id

            elif source_node_search_result and destination_node_search_result:
                cypher = f"""
                MATCH (source)
                WHERE elementId(source) = $source_id
                SET source.mentions = coalesce(source.mentions, 0) + 1
                WITH source
                MATCH (destination)
                WHERE elementId(destination) = $destination_id
                SET destination.mentions = coalesce(destination.mentions, 0) + 1
                MERGE (source)-[r:{relationship}]->(destination)
                ON CREATE SET 
                    r.created_at = timestamp(),
                    r.updated_at = timestamp(),
                    r.mentions = 1
                ON MATCH SET r.mentions = coalesce(r.mentions, 0) + 1
                RETURN source.name AS source, type(r) AS relationship, destination.name AS target
                """

                params = {
                    "source_id": source_node_search_result[0]["elementId(source_candidate)"],
                    "destination_id": destination_node_search_result[0]["elementId(destination_candidate)"],
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id

            else:
                # Build dynamic MERGE props for both source and destination
                source_props = ["name: $source_name", "user_id: $user_id"]
                dest_props = ["name: $dest_name", "user_id: $user_id"]
                if agent_id:
                    source_props.append("agent_id: $agent_id")
                    dest_props.append("agent_id: $agent_id")
                if run_id:
                    source_props.append("run_id: $run_id")
                    dest_props.append("run_id: $run_id")
                source_props_str = ", ".join(source_props)
                dest_props_str = ", ".join(dest_props)

                cypher = f"""
                MERGE (source {source_label} {{{source_props_str}}})
                ON CREATE SET source.created = timestamp(),
                            source.mentions = 1
                            {source_extra_set}
                ON MATCH SET source.mentions = coalesce(source.mentions, 0) + 1
                WITH source
                CALL db.create.setNodeVectorProperty(source, 'embedding', $source_embedding)
                WITH source
                MERGE (destination {destination_label} {{{dest_props_str}}})
                ON CREATE SET destination.created = timestamp(),
                            destination.mentions = 1
                            {destination_extra_set}
                ON MATCH SET destination.mentions = coalesce(destination.mentions, 0) + 1
                WITH source, destination
                CALL db.create.setNodeVectorProperty(destination, 'embedding', $dest_embedding)
                WITH source, destination
                MERGE (source)-[rel:{relationship}]->(destination)
                ON CREATE SET rel.created = timestamp(), rel.mentions = 1
                ON MATCH SET rel.mentions = coalesce(rel.mentions, 0) + 1
                RETURN source.name AS source, type(rel) AS relationship, destination.name AS target
                """

                params = {
                    "source_name": source,
                    "dest_name": destination,
                    "source_embedding": source_embedding,
                    "dest_embedding": dest_embedding,
                    "user_id": user_id,
                }
                if agent_id:
                    params["agent_id"] = agent_id
                if run_id:
                    params["run_id"] = run_id
            result = self.graph.query(cypher, params=params)
            results.append(result)
        return results

    def _remove_spaces_from_entities(self, entity_list):
        for item in entity_list:
            item["source"] = item["source"].lower().replace(" ", "_")
            # Use the sanitization function for relationships to handle special characters
            item["relationship"] = sanitize_relationship_for_cypher(item["relationship"].lower().replace(" ", "_"))
            item["destination"] = item["destination"].lower().replace(" ", "_")
        return entity_list

    def _search_source_node(self, source_embedding, filters, threshold=0.9):
        # Build WHERE conditions
        where_conditions = ["source_candidate.embedding IS NOT NULL", "source_candidate.user_id = $user_id"]
        if filters.get("agent_id"):
            where_conditions.append("source_candidate.agent_id = $agent_id")
        if filters.get("run_id"):
            where_conditions.append("source_candidate.run_id = $run_id")
        where_clause = " AND ".join(where_conditions)

        cypher = f"""
            MATCH (source_candidate {self.node_label})
            WHERE {where_clause}

            WITH source_candidate,
            round(2 * vector.similarity.cosine(source_candidate.embedding, $source_embedding) - 1, 4) AS source_similarity // denormalize for backward compatibility
            WHERE source_similarity >= $threshold

            WITH source_candidate, source_similarity
            ORDER BY source_similarity DESC
            LIMIT 1

            RETURN elementId(source_candidate)
            """

        params = {
            "source_embedding": source_embedding,
            "user_id": filters["user_id"],
            "threshold": threshold,
        }
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]

        result = self.graph.query(cypher, params=params)
        return result

    def _search_destination_node(self, destination_embedding, filters, threshold=0.9):
        # Build WHERE conditions
        where_conditions = ["destination_candidate.embedding IS NOT NULL", "destination_candidate.user_id = $user_id"]
        if filters.get("agent_id"):
            where_conditions.append("destination_candidate.agent_id = $agent_id")
        if filters.get("run_id"):
            where_conditions.append("destination_candidate.run_id = $run_id")
        where_clause = " AND ".join(where_conditions)

        cypher = f"""
            MATCH (destination_candidate {self.node_label})
            WHERE {where_clause}

            WITH destination_candidate,
            round(2 * vector.similarity.cosine(destination_candidate.embedding, $destination_embedding) - 1, 4) AS destination_similarity // denormalize for backward compatibility

            WHERE destination_similarity >= $threshold

            WITH destination_candidate, destination_similarity
            ORDER BY destination_similarity DESC
            LIMIT 1

            RETURN elementId(destination_candidate)
            """

        params = {
            "destination_embedding": destination_embedding,
            "user_id": filters["user_id"],
            "threshold": threshold,
        }
        if filters.get("agent_id"):
            params["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            params["run_id"] = filters["run_id"]

        result = self.graph.query(cypher, params=params)
        return result

    # Reset is not defined in base.py
    def reset(self):
        """Reset the graph by clearing all nodes and relationships."""
        logger.warning("Clearing graph...")
        cypher_query = """
        MATCH (n) DETACH DELETE n
        """
        return self.graph.query(cypher_query)
