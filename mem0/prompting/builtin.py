import json
from typing import Dict

from mem0.configs.prompts import (
    AGENT_MEMORY_EXTRACTION_PROMPT,
    DEFAULT_UPDATE_MEMORY_PROMPT,
    MEMORY_ANSWER_PROMPT,
    PROCEDURAL_MEMORY_SYSTEM_PROMPT,
    get_user_memory_extraction_prompt,
)
from mem0.graphs.utils import DELETE_RELATIONS_SYSTEM_PROMPT, EXTRACT_RELATIONS_PROMPT
from mem0.reranker.llm_reranker import LLMReranker


ENTITY_EXTRACTION_TEMPLATE = """
### Task
Extract entities from the text and classify them into EXACTLY ONE of the 4 predefined categories below.

### Entity Type Categories (MUST use one of these)
1. **person** - Individual people, social roles, or the user themselves
2. **place** - Geographic locations, venues, buildings, or points of interest
3. **object** - All other entities including food, organizations, events, apps, products, projects
4. **category** - Abstract concepts, interests, topics, or fields of study

### Special Rules
1. Self-Reference (I, me, my, myself) -> entity: "{{ user_id }}", type: "person"
2. User mentioned by name -> entity: "{{ user_id }}", type: "person"
3. If input is a QUESTION, DO NOT answer it. Only extract entities mentioned.
4. Each entity MUST have ONE of the 4 types above. NO OTHER types allowed.
5. DO NOT extract verbs or actions as entities.
6. DO NOT extract time expressions, dates, or temporal words as entities.

### Examples
Input: "My boss and I discussed Project X at a cafe in Barcelona."
Output:
- {{ user_id }} | person
- boss | person
- project_x | object
- cafe | place
- barcelona | place

Input: "I love photography, sunset views, and eating sushi."
Output:
- {{ user_id }} | person
- photography | category
- sunset | category
- sushi | object

Input: "I posted a photo on Instagram about the company retreat."
Output:
- {{ user_id }} | person
- photo | object
- instagram | object
- company_retreat | object

Input: "What is my doctor's appointment at the hospital schedule?"
Output:
- {{ user_id }} | person
- doctor | person
- hospital | object

Input: "I had coffee at Starbucks near my office."
Output:
- {{ user_id }} | person
- coffee | object
- starbucks | object
- office | place

Now extract entities from the text above, using ONLY the 4 allowed types.
""".strip()


BUILTIN_PROMPTS: Dict[str, str] = {
    "semantic.user_fact.system": get_user_memory_extraction_prompt("{{ user_id }}"),
    "semantic.user_fact.user": "Current request user_id: {{ user_id }}\nInput:\n{{ parsed_messages }}",
    "semantic.agent_fact.system": AGENT_MEMORY_EXTRACTION_PROMPT,
    "semantic.agent_fact.user": "Input:\n{{ parsed_messages }}",
    "semantic.update_memory.prefix": DEFAULT_UPDATE_MEMORY_PROMPT,
    "memory.answer.system": MEMORY_ANSWER_PROMPT,
    "memory.procedural.system": PROCEDURAL_MEMORY_SYSTEM_PROMPT,
    "graph.entity_extraction.system": ENTITY_EXTRACTION_TEMPLATE,
    "graph.relation.system": EXTRACT_RELATIONS_PROMPT.replace("USER_ID", "{{ user_identity }}").replace(
        "CUSTOM_PROMPT", "{{ custom_prompt_line }}"
    ),
    "graph.relation.user_with_entities": "List of entities: {{ entity_list }}.\n\nText: {{ data }}",
    "graph.relation.user_raw": "{{ data }}",
    "graph.delete.system": DELETE_RELATIONS_SYSTEM_PROMPT.replace("USER_ID", "{{ user_identity }}"),
    "graph.delete.user": "Here are the existing memories: {{ existing_memories_string }} \n\n New Information: {{ data }}",
    "summary.execution.system": PROCEDURAL_MEMORY_SYSTEM_PROMPT,
    "reranker.llm.default": LLMReranker._get_default_prompt(LLMReranker),
}


def normalize_prompt_payload(payload: object) -> Dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError("Prompt payload must be a dictionary of prompt_key -> template")
    normalized: Dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            normalized[key] = value
        else:
            normalized[key] = json.dumps(value, ensure_ascii=True)
    return normalized
