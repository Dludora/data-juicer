from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

from loguru import logger

from data_juicer.core.metadata.models import Ctx, OpCtx
from data_juicer.utils.config_utils import ConfigAccessor

ENTRY_POINT_GROUP = "data_juicer.metadata_providers"


class MetadataProvider:
    name = "metadata_provider"

    def __init__(self, executor: Any, cfg: Any, executor_type: str, provider_cfg: dict[str, Any] | None = None):
        self.executor = executor
        self.cfg = cfg
        self.executor_type = executor_type
        self.provider_cfg = provider_cfg or {}

    def is_enabled(self) -> bool:
        return True

    def on_pipeline_started(self, ctx: Ctx) -> None:
        return None

    def on_pipeline_completed(self, ctx: Ctx) -> None:
        return None

    def on_pipeline_failed(self, ctx: Ctx, error: Exception) -> None:
        return None

    def on_operator_started(self, ctx: Ctx, op_ctx: OpCtx) -> None:
        return None

    def on_operator_completed(self, ctx: Ctx, op_ctx: OpCtx) -> None:
        return None

    def on_operator_failed(self, ctx: Ctx, op_ctx: OpCtx, error: Exception) -> None:
        return None


class EventLogProvider(MetadataProvider):
    name = "event_log"

    def is_enabled(self) -> bool:
        return getattr(self.executor, "event_logger", None) is not None

    def on_pipeline_started(self, ctx: Ctx) -> None:
        self.executor.log_job_start(ctx.extra.get("job_config", {}), ctx.extra.get("num_operators", 0))

    def on_pipeline_completed(self, ctx: Ctx) -> None:
        duration = self._ctx_duration(ctx)
        output_path = ConfigAccessor.get(self.cfg, "export_path", None)
        self.executor.log_job_complete(duration, output_path)

    def on_pipeline_failed(self, ctx: Ctx, error: Exception) -> None:
        duration = self._ctx_duration(ctx)
        self.executor.log_job_failed(str(error), duration)

    def on_operator_started(self, ctx: Ctx, op_ctx: OpCtx) -> None:
        self.executor.log_op_start(
            op_ctx.partition_id,
            op_ctx.op_name,
            op_ctx.op_index,
            op_ctx.extra.get("op_args", {}),
            metadata=self._op_metadata(op_ctx),
        )

    def on_operator_completed(self, ctx: Ctx, op_ctx: OpCtx) -> None:
        metrics = op_ctx.metrics or {}
        self.executor.log_op_complete(
            op_ctx.partition_id,
            op_ctx.op_name,
            op_ctx.op_index,
            metrics.get("duration_seconds", self._op_duration(op_ctx)),
            metrics.get("checkpoint_path"),
            metrics.get("input_rows"),
            metrics.get("output_rows"),
            metadata=self._op_metadata(op_ctx),
        )

    def on_operator_failed(self, ctx: Ctx, op_ctx: OpCtx, error: Exception) -> None:
        retry_count = (op_ctx.metrics or {}).get("retry_count", 0)
        self.executor.log_op_failed(
            op_ctx.partition_id,
            op_ctx.op_name,
            op_ctx.op_index,
            str(error),
            retry_count,
            metadata=self._op_metadata(op_ctx),
        )

    @staticmethod
    def _ctx_duration(ctx: Ctx) -> float:
        if not ctx.started_at or not ctx.ended_at:
            return 0.0
        try:
            from datetime import datetime

            return max(
                0.0, (datetime.fromisoformat(ctx.ended_at) - datetime.fromisoformat(ctx.started_at)).total_seconds()
            )
        except Exception:
            return 0.0

    @staticmethod
    def _op_duration(op_ctx: OpCtx) -> float:
        if not op_ctx.started_at or not op_ctx.ended_at:
            return 0.0
        try:
            from datetime import datetime

            return max(
                0.0,
                (datetime.fromisoformat(op_ctx.ended_at) - datetime.fromisoformat(op_ctx.started_at)).total_seconds(),
            )
        except Exception:
            return 0.0

    @staticmethod
    def _op_metadata(op_ctx: OpCtx) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        dag_info = op_ctx.extra.get("dag", {})
        if dag_info.get("node_id"):
            metadata["dag_node_id"] = dag_info["node_id"]
        if op_ctx.extra.get("storage_kind"):
            metadata["storage_kind"] = op_ctx.extra["storage_kind"]
        return metadata


def load_metadata_providers(executor: Any, cfg: Any, executor_type: str) -> list[MetadataProvider]:
    metadata_cfg = ConfigAccessor.get(cfg, "metadata", {})
    if not ConfigAccessor.get(metadata_cfg, "enabled", True):
        return []

    providers: list[MetadataProvider] = []
    provider_configs = _normalize_provider_configs(ConfigAccessor.get(metadata_cfg, "providers", None))
    if not provider_configs:
        provider_configs = [{"name": EventLogProvider.name, "enabled": True}]

    for item in provider_configs:
        name = item.get("name")
        enabled = item.get("enabled", True)
        provider_cfg = item.get("config", {}) or {}
        if not enabled:
            continue
        if name == EventLogProvider.name:
            provider = EventLogProvider(executor, cfg, executor_type, provider_cfg)
            if provider.is_enabled():
                providers.append(provider)
            continue
        provider = _load_external_provider(name, executor, cfg, executor_type, provider_cfg)
        if provider is not None and provider.is_enabled():
            providers.append(provider)
    return providers


def _normalize_provider_configs(config: Any) -> list[dict[str, Any]]:
    if not config:
        return []
    if isinstance(config, list):
        return [item for item in config if isinstance(item, dict)]
    if isinstance(config, dict):
        return [config]
    return []


def _load_external_provider(
    name: str,
    executor: Any,
    cfg: Any,
    executor_type: str,
    provider_cfg: dict[str, Any],
) -> MetadataProvider | None:
    try:
        available = entry_points(group=ENTRY_POINT_GROUP)
    except Exception as e:
        logger.warning(f"Failed to inspect metadata provider entry points: {e}")
        return None

    for entry_point in available:
        if entry_point.name != name:
            continue
        try:
            provider_cls = entry_point.load()
            provider = provider_cls(executor, cfg, executor_type, provider_cfg)
            if not isinstance(provider, MetadataProvider) and not hasattr(provider, "on_pipeline_started"):
                logger.warning(f"Metadata provider [{name}] does not implement the expected interface")
                return None
            return provider
        except Exception as e:
            logger.warning(f"Failed to load metadata provider [{name}]: {e}")
            return None

    logger.warning(f"Metadata provider [{name}] is not registered")
    return None
