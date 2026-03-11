from typing import Any, Dict, Optional

from mem0.prompting.renderer import render_prompt
from mem0.prompting.sources import (
    DEFAULT_PROMPT_SOURCE_REGISTRY,
    BuiltinPromptSource,
    PromptSnapshot,
    PromptSource,
    PromptSourceRegistry,
    resolve_prompt_source,
)


class PromptRegistry:
    def __init__(
        self,
        builtin_source: Optional[PromptSource] = None,
        external_source: Optional[PromptSource] = None,
        overrides: Optional[Dict[str, str]] = None,
    ):
        self._builtin_source = builtin_source or BuiltinPromptSource()
        self._external_source = external_source
        self._overrides = overrides or {}
        self._fingerprint: Optional[str] = None
        self._snapshot = self._compose_snapshot()

    @classmethod
    def from_config(
        cls,
        prompting_config: Any,
        source_registry: Optional[PromptSourceRegistry] = None,
    ) -> "PromptRegistry":
        if prompting_config is None:
            return cls()

        overrides = getattr(prompting_config, "overrides", None) or {}
        external_source = resolve_prompt_source(prompting_config, source_registry or DEFAULT_PROMPT_SOURCE_REGISTRY)
        return cls(external_source=external_source, overrides=overrides)

    def _compose_snapshot(self) -> PromptSnapshot:
        builtin_snapshot = self._builtin_source.load()
        prompts = dict(builtin_snapshot.prompts)
        source = builtin_snapshot.source
        version = builtin_snapshot.version

        if self._external_source:
            external_snapshot = self._external_source.load()
            prompts.update(external_snapshot.prompts)
            source = external_snapshot.source
            version = external_snapshot.version

        prompts.update(self._overrides)
        if self._overrides:
            version = f"{version}-overrides"
        return PromptSnapshot(prompts=prompts, version=version, source=source)

    def reload_if_needed(self) -> PromptSnapshot:
        if not self._external_source:
            return self._snapshot

        fingerprint = self._external_source.fingerprint()
        if fingerprint != self._fingerprint:
            self._snapshot = self._compose_snapshot()
            self._fingerprint = fingerprint
        return self._snapshot

    def snapshot(self) -> PromptSnapshot:
        return self.reload_if_needed()

    def get_template(self, key: str) -> str:
        snapshot = self.reload_if_needed()
        if key not in snapshot.prompts:
            raise KeyError(f"Unknown prompt key: {key}")
        return snapshot.prompts[key]

    def render(self, key: str, context: Dict[str, Any]) -> str:
        return render_prompt(self.get_template(key), context)
