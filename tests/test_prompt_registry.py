import json
from types import SimpleNamespace

from mem0.memory.main import Memory
from mem0.prompting import PromptRegistry, PromptSource, PromptSourceRegistry


class InlinePromptSource(PromptSource):
    def __init__(self, prompts):
        self.prompts = prompts

    def load(self):
        return SimpleNamespace(prompts=self.prompts, version="inline-v1", source="inline")

    def fingerprint(self):
        return "inline"


def build_inline_source(prompting_config):
    return InlinePromptSource(prompting_config.source_config["prompts"])


def test_prompt_registry_reloads_external_snapshot(tmp_path):
    prompt_file = tmp_path / "prompts.json"
    prompt_file.write_text(json.dumps({"semantic.user_fact.system": "first {{ user_id }}"}), encoding="utf-8")

    registry = PromptRegistry.from_config(
        SimpleNamespace(source_path=str(prompt_file), source_format="json", overrides={})
    )

    first_snapshot = registry.snapshot()
    assert registry.render("semantic.user_fact.system", {"user_id": "alice"}) == "first alice"

    prompt_file.write_text(json.dumps({"semantic.user_fact.system": "second {{ user_id }}"}), encoding="utf-8")

    second_snapshot = registry.reload_if_needed()
    assert second_snapshot.version != first_snapshot.version
    assert registry.render("semantic.user_fact.system", {"user_id": "bob"}) == "second bob"


def test_prompt_registry_supports_registered_source_loader():
    source_registry = PromptSourceRegistry()
    source_registry.register("inline", build_inline_source)

    registry = PromptRegistry.from_config(
        SimpleNamespace(
            source_loader="inline",
            source_path=None,
            source_format=None,
            source_config={"prompts": {"semantic.user_fact.system": "inline {{ user_id }}"}},
            overrides={},
        ),
        source_registry=source_registry,
    )

    assert registry.render("semantic.user_fact.system", {"user_id": "carol"}) == "inline carol"


def test_memory_registration_helper_registers_prompt_source_loader():
    Memory.register_prompt_source_loader("inline_via_memory", build_inline_source)

    registry = PromptRegistry.from_config(
        SimpleNamespace(
            source_loader="inline_via_memory",
            source_path=None,
            source_format=None,
            source_config={"prompts": {"semantic.user_fact.system": "memory {{ user_id }}"}},
            overrides={},
        )
    )

    assert registry.render("semantic.user_fact.system", {"user_id": "dave"}) == "memory dave"
