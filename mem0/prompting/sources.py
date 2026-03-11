import hashlib
import importlib
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from mem0.prompting.builtin import BUILTIN_PROMPTS, normalize_prompt_payload


@dataclass
class PromptSnapshot:
    prompts: Dict[str, str]
    version: str
    source: str


class PromptSource:
    def load(self) -> PromptSnapshot:
        raise NotImplementedError

    def fingerprint(self) -> str:
        raise NotImplementedError


class BuiltinPromptSource(PromptSource):
    def load(self) -> PromptSnapshot:
        prompts = dict(BUILTIN_PROMPTS)
        version = hashlib.md5(json.dumps(prompts, sort_keys=True).encode()).hexdigest()
        return PromptSnapshot(prompts=prompts, version=version, source="builtin")

    def fingerprint(self) -> str:
        return "builtin"


class JsonFilePromptSource(PromptSource):
    def __init__(self, path: str):
        self.path = path

    def load(self) -> PromptSnapshot:
        with open(self.path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        prompts = normalize_prompt_payload(payload)
        version = hashlib.md5(json.dumps(prompts, sort_keys=True).encode()).hexdigest()
        return PromptSnapshot(prompts=prompts, version=version, source=self.path)

    def fingerprint(self) -> str:
        stat = os.stat(self.path)
        return f"{self.path}:{stat.st_mtime_ns}:{stat.st_size}"


class YamlFilePromptSource(PromptSource):
    def __init__(self, path: str):
        self.path = path

    def load(self) -> PromptSnapshot:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to load YAML prompt files.") from exc

        with open(self.path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        prompts = normalize_prompt_payload(payload)
        version = hashlib.md5(json.dumps(prompts, sort_keys=True).encode()).hexdigest()
        return PromptSnapshot(prompts=prompts, version=version, source=self.path)

    def fingerprint(self) -> str:
        stat = os.stat(self.path)
        return f"{self.path}:{stat.st_mtime_ns}:{stat.st_size}"


def build_file_prompt_source(path: Optional[str], fmt: Optional[str]) -> Optional[PromptSource]:
    if not path:
        return None

    if fmt:
        normalized = fmt.lower()
    else:
        _, ext = os.path.splitext(path)
        normalized = ext.lstrip(".").lower()

    if normalized == "json":
        return JsonFilePromptSource(path)
    if normalized in {"yaml", "yml"}:
        return YamlFilePromptSource(path)
    raise ValueError(f"Unsupported prompt source format: {normalized}")


PromptSourceBuilder = Callable[[Any], PromptSource]


class PromptSourceRegistry:
    def __init__(self):
        self._builders: Dict[str, PromptSourceBuilder] = {
            "file": self._build_file_source,
            "json_file": self._build_json_source,
            "yaml_file": self._build_yaml_source,
        }

    def register(self, name: str, builder: PromptSourceBuilder) -> None:
        self._builders[name] = builder

    def create(self, prompting_config: Any) -> Optional[PromptSource]:
        if prompting_config is None:
            return None

        loader_name = getattr(prompting_config, "source_loader", None)
        source_path = getattr(prompting_config, "source_path", None)
        source_config = getattr(prompting_config, "source_config", None) or {}

        if loader_name is None and not source_path:
            return None

        if loader_name is None:
            loader_name = "file"

        builder = self._builders.get(loader_name)
        if builder is None:
            builder = self._import_builder(loader_name)

        source = builder(prompting_config)
        if not isinstance(source, PromptSource):
            raise TypeError(f"Prompt source loader {loader_name!r} must return a PromptSource instance")

        if source_config and hasattr(source, "config"):
            setattr(source, "config", source_config)
        return source

    @staticmethod
    def _build_file_source(prompting_config: Any) -> Optional[PromptSource]:
        return build_file_prompt_source(
            getattr(prompting_config, "source_path", None),
            getattr(prompting_config, "source_format", None),
        )

    @staticmethod
    def _build_json_source(prompting_config: Any) -> PromptSource:
        path = getattr(prompting_config, "source_path", None)
        if not path:
            raise ValueError("prompting.source_path is required for json_file prompt source")
        return JsonFilePromptSource(path)

    @staticmethod
    def _build_yaml_source(prompting_config: Any) -> PromptSource:
        path = getattr(prompting_config, "source_path", None)
        if not path:
            raise ValueError("prompting.source_path is required for yaml_file prompt source")
        return YamlFilePromptSource(path)

    @staticmethod
    def _import_builder(import_path: str) -> PromptSourceBuilder:
        module_path, separator, attr_name = import_path.replace(":", ".").rpartition(".")
        if not separator:
            raise KeyError(f"Unknown prompt source loader: {import_path}")
        module = importlib.import_module(module_path)
        builder = getattr(module, attr_name)
        if not callable(builder):
            raise TypeError(f"Prompt source loader {import_path!r} is not callable")
        return builder


DEFAULT_PROMPT_SOURCE_REGISTRY = PromptSourceRegistry()


def resolve_prompt_source(prompting_config: Any, registry: Optional[PromptSourceRegistry] = None) -> Optional[PromptSource]:
    registry = registry or DEFAULT_PROMPT_SOURCE_REGISTRY
    return registry.create(prompting_config)
