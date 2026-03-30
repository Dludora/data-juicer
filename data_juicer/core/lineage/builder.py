import hashlib
import json
import os
import socket
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

from data_juicer import __version__
from data_juicer.core.lineage.models import DatasetRef, PipelineRunContext, RecipeInfo
from data_juicer.utils.config_utils import ConfigAccessor

DEFAULT_PRODUCER = f"https://github.com/modelscope/data-juicer/tree/v{__version__}"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    return ConfigAccessor.get(cfg, key, default)


def _dataset_ref_from_uri(uri: str, role: str) -> DatasetRef:
    parsed = urlparse(uri)
    if parsed.scheme:
        namespace = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else parsed.scheme
        name = parsed.path.lstrip("/") or parsed.netloc or "unknown"
        platform = parsed.scheme
    else:
        abs_path = os.path.abspath(uri)
        namespace = f"file://{socket.gethostname()}"
        name = abs_path
        platform = "filesystem"

    return DatasetRef(
        namespace=namespace,
        name=name,
        uri=uri,
        role=role,
        platform=platform,
    )


def _extract_dataset_path_tokens(dataset_path: str) -> Iterable[str]:
    if not dataset_path:
        return []
    tokens = dataset_path.split()
    resolved: List[str] = []
    for token in tokens:
        try:
            weight = float(token)
            if 0.0 <= weight <= 1.0:
                continue
        except ValueError:
            pass
        resolved.append(token)
    return resolved


def _extract_inputs_from_dataset_cfg(dataset_cfg: Any) -> List[str]:
    if not dataset_cfg:
        return []

    if isinstance(dataset_cfg, list):
        cfgs = dataset_cfg
    elif isinstance(dataset_cfg, dict):
        cfgs = dataset_cfg.get("configs", []) or [dataset_cfg]
    else:
        cfgs = ConfigAccessor.get(dataset_cfg, "configs", [])
        if not cfgs:
            cfgs = [dataset_cfg]

    uris = []
    for item in cfgs:
        path = ConfigAccessor.get(item, "path")
        if path:
            uris.append(path)
            continue
        uri = ConfigAccessor.get(item, "uri")
        if uri:
            uris.append(uri)
            continue
        hf_path = ConfigAccessor.get(item, "dataset_path")
        if hf_path:
            uris.append(hf_path)
    return uris


def _build_input_refs(cfg: Any) -> List[DatasetRef]:
    refs: List[DatasetRef] = []

    dataset_path = _cfg_value(cfg, "dataset_path", "")
    for token in _extract_dataset_path_tokens(dataset_path):
        refs.append(_dataset_ref_from_uri(token, role="input"))

    dataset_cfg = _cfg_value(cfg, "dataset", None)
    for uri in _extract_inputs_from_dataset_cfg(dataset_cfg):
        refs.append(_dataset_ref_from_uri(uri, role="input"))

    deduped: Dict[str, DatasetRef] = {}
    for ref in refs:
        deduped[f"{ref.namespace}:{ref.name}"] = ref
    return list(deduped.values())


def _build_output_refs(output_path: Optional[str]) -> List[DatasetRef]:
    if not output_path:
        return []
    return [_dataset_ref_from_uri(output_path, role="output")]


def _build_recipe_info(cfg: Any) -> RecipeInfo:
    process = _cfg_value(cfg, "process", []) or []
    payload = json.dumps(process, sort_keys=True, ensure_ascii=False)
    process_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    return RecipeInfo(
        project_name=_cfg_value(cfg, "project_name", "data_juicer"),
        executor_type=_cfg_value(cfg, "executor_type", "default"),
        config_path=_cfg_value(cfg, "config", None),
        process=process,
        process_hash=process_hash,
        num_operators=len(process),
    )


def _build_job_name(cfg: Any, executor_type: str) -> str:
    lineage_cfg = _cfg_value(cfg, "lineage", {})
    template = ConfigAccessor.get(lineage_cfg, "job_name_template", "{project_name}")
    project_name = _cfg_value(cfg, "project_name", "data_juicer")
    job_id = _cfg_value(cfg, "job_id", "")
    try:
        return template.format(project_name=project_name, executor_type=executor_type, job_id=job_id)
    except Exception:
        return project_name


def build_pipeline_context(
    cfg: Any,
    executor_type: str,
    event_type: str,
    run_id: str,
    status: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    error_message: Optional[str] = None,
    output_path: Optional[str] = None,
    extra_run: Optional[Dict[str, Any]] = None,
    extra_job: Optional[Dict[str, Any]] = None,
) -> PipelineRunContext:
    lineage_cfg = _cfg_value(cfg, "lineage", {})
    namespace = ConfigAccessor.get(lineage_cfg, "namespace", None) or f"data_juicer.{executor_type}"
    producer = ConfigAccessor.get(lineage_cfg, "producer", None) or DEFAULT_PRODUCER

    return PipelineRunContext(
        event_type=event_type,
        event_time=_utc_now_iso(),
        run_id=run_id,
        job_namespace=namespace,
        job_name=_build_job_name(cfg, executor_type),
        producer=producer,
        job_id=_cfg_value(cfg, "job_id", None),
        status=status,
        duration_seconds=duration_seconds,
        error_message=error_message,
        inputs=_build_input_refs(cfg),
        outputs=_build_output_refs(output_path or _cfg_value(cfg, "export_path", None)),
        recipe=_build_recipe_info(cfg),
        extra_run=extra_run or {},
        extra_job=extra_job or {},
    )
