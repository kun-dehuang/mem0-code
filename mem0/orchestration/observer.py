import logging
import importlib
from typing import Any, Callable, Dict, List, Optional

from mem0.storage.pipeline_state import SQLitePipelineStateStore


logger = logging.getLogger(__name__)


class PipelineObserver:
    def emit(self, job_id: str, stage_name: str, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError


class LoggerPipelineObserver(PipelineObserver):
    def emit(self, job_id: str, stage_name: str, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        logger.info(
            "mem0.pipeline job_id=%s stage=%s event=%s payload=%s",
            job_id,
            stage_name,
            event_type,
            payload or {},
        )


class InMemoryPipelineObserver(PipelineObserver):
    def __init__(self):
        self.events: List[Dict[str, Any]] = []

    def emit(self, job_id: str, stage_name: str, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        self.events.append(
            {
                "job_id": job_id,
                "stage_name": stage_name,
                "event_type": event_type,
                "payload": payload or {},
            }
        )


class DurablePipelineObserver(PipelineObserver):
    def __init__(self, state_store: SQLitePipelineStateStore):
        self.state_store = state_store

    def emit(self, job_id: str, stage_name: str, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        self.state_store.append_event(job_id, stage_name, event_type, payload)


class MultiPipelineObserver(PipelineObserver):
    def __init__(self, observers: List[PipelineObserver]):
        self.observers = observers

    def emit(self, job_id: str, stage_name: str, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        for observer in self.observers:
            observer.emit(job_id, stage_name, event_type, payload)


ObserverBuilder = Callable[[SQLitePipelineStateStore, Dict[str, Any]], PipelineObserver]


class PipelineObserverRegistry:
    def __init__(self):
        self._builders: Dict[str, ObserverBuilder] = {
            "logger": lambda state_store, config: LoggerPipelineObserver(),
            "durable": lambda state_store, config: DurablePipelineObserver(state_store),
            "in_memory": lambda state_store, config: InMemoryPipelineObserver(),
        }

    def register(self, name: str, builder: ObserverBuilder) -> None:
        self._builders[name] = builder

    def create(self, name: str, state_store: SQLitePipelineStateStore, sink_config: Optional[Dict[str, Any]] = None) -> PipelineObserver:
        builder = self._builders.get(name)
        if builder is None:
            builder = self._import_builder(name)
        observer = builder(state_store, sink_config or {})
        if not isinstance(observer, PipelineObserver):
            raise TypeError(f"Pipeline observer sink {name!r} must return a PipelineObserver")
        return observer

    @staticmethod
    def _import_builder(import_path: str) -> ObserverBuilder:
        module_path, separator, attr_name = import_path.replace(":", ".").rpartition(".")
        if not separator:
            raise KeyError(f"Unknown observer sink: {import_path}")
        module = importlib.import_module(module_path)
        builder = getattr(module, attr_name)
        if not callable(builder):
            raise TypeError(f"Observer builder {import_path!r} is not callable")
        return builder


DEFAULT_PIPELINE_OBSERVER_REGISTRY = PipelineObserverRegistry()


def build_pipeline_observer(
    config: Any,
    state_store: SQLitePipelineStateStore,
    registry: Optional[PipelineObserverRegistry] = None,
) -> PipelineObserver:
    observers: List[PipelineObserver] = []
    registry = registry or DEFAULT_PIPELINE_OBSERVER_REGISTRY
    observability = getattr(config, "observability", None)
    enable_logger = True if observability is None else observability.enable_logger_sink
    enable_durable = False if observability is None else observability.enable_durable_sink
    enable_in_memory = False if observability is None else observability.enable_in_memory_sink
    custom_sinks = [] if observability is None else list(getattr(observability, "sinks", []) or [])
    sink_configs = {} if observability is None else dict(getattr(observability, "sink_configs", {}) or {})

    if enable_logger:
        observers.append(registry.create("logger", state_store, sink_configs.get("logger")))
    if enable_durable:
        observers.append(registry.create("durable", state_store, sink_configs.get("durable")))
    if enable_in_memory:
        observers.append(registry.create("in_memory", state_store, sink_configs.get("in_memory")))
    for sink_name in custom_sinks:
        observers.append(registry.create(sink_name, state_store, sink_configs.get(sink_name)))
    if not observers:
        observers.append(registry.create("logger", state_store, sink_configs.get("logger")))
    return MultiPipelineObserver(observers)
