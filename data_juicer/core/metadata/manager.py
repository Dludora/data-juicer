from __future__ import annotations

import hashlib
import inspect
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import git
from loguru import logger

from data_juicer.core.metadata.models import Ctx, OpCtx, OpKey
from data_juicer.core.metadata.provider import MetadataProvider, load_metadata_providers
from data_juicer.core.metadata.resolver import MetadataResolver
from data_juicer.utils.config_utils import ConfigAccessor


class MetadataManager:
    def __init__(
        self,
        executor: Any,
        cfg: Any,
        executor_type: str,
        providers: list[MetadataProvider] | None = None,
    ):
        self.executor = executor
        self.cfg = cfg
        self.executor_type = executor_type
        self.resolver = MetadataResolver(cfg)
        self.providers = providers if providers is not None else load_metadata_providers(executor, cfg, executor_type)
        metadata_cfg = ConfigAccessor.get(cfg, "metadata", {})
        self.enabled = bool(ConfigAccessor.get(metadata_cfg, "enabled", True))
        self.ctx: Ctx | None = None
        self._op_source_cache: dict[type, dict[str, Any]] = {}

    def start_pipeline(
        self,
        input_dataset_obj: Any | None,
        operators: list[Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Ctx | None:
        if not self.enabled:
            return None

        operator_names = [getattr(op, "_name", op.__class__.__name__) for op in (operators or [])]
        pipeline_recipe = self._build_pipeline_recipe()
        ctx_extra = {
            "recipe": pipeline_recipe,
            "recipe_hash": self._hash_recipe(pipeline_recipe),
            "num_operators": len(operators or []),
            "operator_names": operator_names,
            "job_config": self._build_job_config(),
            "dag": self._build_dag_metadata(),
            "work_dir": getattr(self.executor, "work_dir", None),
            "config_path": self._resolve_config_path(),
        }
        if extra:
            ctx_extra.update(extra)

        input_snapshot = self.resolver.build_pipeline_input_snapshot(input_dataset_obj)
        self.ctx = Ctx(
            run_id=self._build_run_id(),
            job_id=ConfigAccessor.get(self.cfg, "job_id", None),
            executor_type=self.executor_type,
            job_name=self._build_job_name(),
            namespace=self._build_namespace(),
            project_name=ConfigAccessor.get(self.cfg, "project_name", "data_juicer"),
            started_at=self._now_iso(),
            input_snapshot=input_snapshot,
            extra=ctx_extra,
        )
        if input_snapshot is not None:
            self.ctx.latest_snapshot_by_partition[0] = input_snapshot
        self._fanout("on_pipeline_started", self.ctx)
        return self.ctx

    def complete_pipeline(self, output_dataset_obj: Any | None) -> None:
        if not self.enabled or self.ctx is None:
            return

        self.ctx.status = "completed"
        self.ctx.ended_at = self._now_iso()
        self.ctx.output_snapshot = self.resolver.build_pipeline_output_snapshot(output_dataset_obj)
        self._fanout("on_pipeline_completed", self.ctx)

    def fail_pipeline(self, error: Exception, output_dataset_obj: Any | None = None) -> None:
        if not self.enabled or self.ctx is None:
            return

        self.ctx.status = "failed"
        self.ctx.ended_at = self._now_iso()
        if output_dataset_obj is not None:
            self.ctx.output_snapshot = self.resolver.build_pipeline_output_snapshot(output_dataset_obj)
        self._fanout("on_pipeline_failed", self.ctx, error)

    def start_operator(
        self,
        op: Any,
        op_index: int,
        input_dataset_obj: Any | None,
        partition_id: int = 0,
    ) -> OpCtx | None:
        if not self.enabled or self.ctx is None:
            return None

        key = OpKey(partition_id=partition_id, op_index=op_index)
        prior_snapshot = self.ctx.latest_snapshot_by_partition.get(partition_id) or self.ctx.input_snapshot
        refs = prior_snapshot.refs if prior_snapshot else None
        storage_kind = prior_snapshot.storage_kind if prior_snapshot else None
        input_snapshot = (
            self.resolver.snapshot(input_dataset_obj, refs=refs, storage_kind=storage_kind) or prior_snapshot
        )
        op_args = dict(getattr(op, "_op_cfg", {}) or {})
        source_info = self._collect_op_source_info(op)
        op_ctx = OpCtx(
            op_id=self._build_op_id(op, op_index, partition_id),
            op_name=getattr(op, "_name", op.__class__.__name__),
            op_type=self._resolve_op_type(op),
            op_index=op_index,
            partition_id=partition_id,
            started_at=self._now_iso(),
            input_snapshot=input_snapshot,
            extra={
                "op_args": op_args,
                "op_config_hash": self._hash_op_config(op_args),
                "storage_kind": input_snapshot.storage_kind if input_snapshot else None,
                "source": source_info,
            },
        )
        dag_node_id = self._get_dag_node_id(op_ctx)
        if dag_node_id:
            op_ctx.extra["dag"] = {"node_id": dag_node_id}
            self._mark_dag_node_started(dag_node_id)
        self.ctx.op_ctxs[key] = op_ctx
        self._fanout("on_operator_started", self.ctx, op_ctx)
        return op_ctx

    def complete_operator(
        self,
        op_index: int,
        output_dataset_obj: Any | None,
        partition_id: int = 0,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or self.ctx is None:
            return

        key = OpKey(partition_id=partition_id, op_index=op_index)
        op_ctx = self.ctx.op_ctxs.get(key)
        if op_ctx is None:
            return

        output_snapshot = self.resolver.build_inmemory_snapshot(
            self.ctx.run_id,
            self._build_op_node_name(op_ctx, "output"),
            output_dataset_obj,
            "output",
        )
        op_ctx.status = "completed"
        op_ctx.ended_at = self._now_iso()
        op_ctx.output_snapshot = output_snapshot
        op_ctx.metrics = dict(metrics or {})
        self.ctx.latest_snapshot_by_partition[partition_id] = output_snapshot

        dag_node_id = ConfigAccessor.get(ConfigAccessor.get(op_ctx.extra, "dag", {}), "node_id", None)
        if dag_node_id:
            duration = float(op_ctx.metrics.get("duration_seconds", 0.0) or 0.0)
            self._mark_dag_node_completed(dag_node_id, duration)
        self._fanout("on_operator_completed", self.ctx, op_ctx)

    def fail_operator(
        self,
        op_index: int,
        error: Exception,
        output_dataset_obj: Any | None = None,
        partition_id: int = 0,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or self.ctx is None:
            return

        key = OpKey(partition_id=partition_id, op_index=op_index)
        op_ctx = self.ctx.op_ctxs.get(key)
        if op_ctx is None:
            return

        op_ctx.status = "failed"
        op_ctx.ended_at = self._now_iso()
        op_ctx.metrics = dict(metrics or {})
        if output_dataset_obj is not None:
            op_ctx.output_snapshot = self.resolver.build_inmemory_snapshot(
                self.ctx.run_id,
                self._build_op_node_name(op_ctx, "failed_output"),
                output_dataset_obj,
                "output",
            )

        dag_node_id = ConfigAccessor.get(ConfigAccessor.get(op_ctx.extra, "dag", {}), "node_id", None)
        if dag_node_id:
            duration = float(op_ctx.metrics.get("duration_seconds", 0.0) or 0.0)
            self._mark_dag_node_failed(dag_node_id, str(error), duration)
        self._fanout("on_operator_failed", self.ctx, op_ctx, error)

    def _fanout(self, method_name: str, *args: Any) -> None:
        for provider in self.providers:
            method = getattr(provider, method_name, None)
            if method is None:
                continue
            try:
                method(*args)
            except Exception as e:
                logger.warning(f"Metadata provider [{provider.name}] failed during [{method_name}]: {e}")

    def _build_job_config(self) -> dict[str, Any]:
        dataset_info: dict[str, Any] = {}
        dataset_path = ConfigAccessor.get(self.cfg, "dataset_path", None)
        dataset_cfg = ConfigAccessor.get(self.cfg, "dataset", None)
        if dataset_path:
            dataset_info["dataset_path"] = dataset_path
        if dataset_cfg:
            dataset_info["dataset"] = dataset_cfg

        job_config = {
            **dataset_info,
            "work_dir": getattr(self.executor, "work_dir", None),
            "executor_type": self.executor_type,
        }
        dag_metadata = self._build_dag_metadata()
        if dag_metadata:
            job_config.update(
                {
                    "dag_node_count": dag_metadata.get("node_count", 0),
                    "dag_edge_count": dag_metadata.get("edge_count", 0),
                    "parallel_groups_count": dag_metadata.get("parallel_groups_count", 0),
                }
            )

        partition_size = getattr(self.executor, "partition_size", None)
        if partition_size is not None:
            job_config["partition_size"] = partition_size
        ckpt_manager = getattr(self.executor, "ckpt_manager", None)
        if ckpt_manager is not None and getattr(ckpt_manager, "checkpoint_strategy", None) is not None:
            strategy = ckpt_manager.checkpoint_strategy
            job_config["checkpoint_strategy"] = getattr(strategy, "value", strategy)
        return job_config

    def _build_dag_metadata(self) -> dict[str, Any]:
        if hasattr(self.executor, "get_dag_metadata"):
            return dict(self.executor.get_dag_metadata() or {})
        pipeline_dag = getattr(self.executor, "pipeline_dag", None)
        if pipeline_dag is None:
            return {"enabled": False}
        return {
            "enabled": True,
            "node_count": len(getattr(pipeline_dag, "nodes", {}) or {}),
            "edge_count": len(getattr(pipeline_dag, "edges", []) or []),
            "parallel_groups_count": len(getattr(pipeline_dag, "parallel_groups", []) or []),
        }

    def _get_dag_node_id(self, op_ctx: OpCtx) -> str | None:
        if hasattr(self.executor, "get_dag_node_id"):
            return self.executor.get_dag_node_id(op_ctx.op_name, op_ctx.op_index, partition_id=op_ctx.partition_id)
        if hasattr(self.executor, "_get_dag_node_for_operation"):
            return self.executor._get_dag_node_for_operation(
                op_ctx.op_name,
                op_ctx.op_index,
                partition_id=op_ctx.partition_id,
            )
        return None

    def _mark_dag_node_started(self, node_id: str) -> None:
        if hasattr(self.executor, "_mark_dag_node_started"):
            self.executor._mark_dag_node_started(node_id)

    def _mark_dag_node_completed(self, node_id: str, duration: float) -> None:
        if hasattr(self.executor, "_mark_dag_node_completed"):
            self.executor._mark_dag_node_completed(node_id, duration)

    def _mark_dag_node_failed(self, node_id: str, error_message: str, duration: float) -> None:
        if hasattr(self.executor, "_mark_dag_node_failed"):
            self.executor._mark_dag_node_failed(node_id, error_message, duration)

    def _build_op_id(self, op: Any, op_index: int, partition_id: int) -> str:
        op_name = getattr(op, "_name", op.__class__.__name__)
        return f"{partition_id}:{op_index:03d}:{op_name}"

    def _build_op_node_name(self, op_ctx: OpCtx, suffix: str) -> str:
        return f"partition_{op_ctx.partition_id}_op_{op_ctx.op_index:03d}_{op_ctx.op_name}_{suffix}"

    def _resolve_op_type(self, op: Any) -> str:
        op_name = op.__class__.__name__.lower()
        if "filter" in op_name:
            return "filter"
        if "mapper" in op_name:
            return "mapper"
        if "deduplicator" in op_name:
            return "deduplicator"
        if "pipeline" in op_name:
            return "pipeline"
        return op.__class__.__name__

    def _build_run_id(self) -> str:
        project_name = ConfigAccessor.get(self.cfg, "project_name", "data_juicer")
        job_id = ConfigAccessor.get(self.cfg, "job_id", None)
        if job_id:
            seed = f"data-juicer:{project_name}:{self.executor_type}:{job_id}"
            return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))
        return str(uuid.uuid4())

    def _build_job_name(self) -> str:
        return ConfigAccessor.get(self.cfg, "project_name", "data_juicer")

    def _build_namespace(self) -> str:
        metadata_cfg = ConfigAccessor.get(self.cfg, "metadata", {})
        return ConfigAccessor.get(metadata_cfg, "namespace", None) or f"data_juicer.{self.executor_type}"

    def _collect_op_source_info(self, op: Any) -> dict[str, Any]:
        op_class = op.__class__
        if op_class in self._op_source_cache:
            return dict(self._op_source_cache[op_class])

        source_info: dict[str, Any] = {
            "module_path": getattr(op_class, "__module__", None),
            "file_path": None,
            "repo_root": None,
            "repo_url": None,
            "repo_owner": None,
            "relative_path": None,
            "git_commit": None,
            "git_tag": None,
            "git_branch": None,
            "git_author_name": None,
            "git_author_email": None,
            "git_committer_name": None,
            "git_committer_email": None,
            "dirty": None,
            "package_version": None,
        }

        try:
            file_path = inspect.getfile(op_class)
            source_info["file_path"] = os.path.abspath(file_path)
        except Exception:
            self._op_source_cache[op_class] = source_info
            return dict(source_info)

        module_path = source_info["module_path"]
        if module_path:
            top_level_module = module_path.split(".", 1)[0]
            try:
                imported = __import__(top_level_module)
                source_info["package_version"] = getattr(imported, "__version__", None)
            except Exception:
                pass

        try:
            repo = git.Repo(source_info["file_path"], search_parent_directories=True)
            source_info["repo_root"] = repo.working_tree_dir
            commit = repo.head.commit
            source_info["git_commit"] = commit.hexsha
            source_info["dirty"] = repo.is_dirty(untracked_files=True)
            source_info["git_author_name"] = getattr(commit.author, "name", None)
            source_info["git_author_email"] = getattr(commit.author, "email", None)
            source_info["git_committer_name"] = getattr(commit.committer, "name", None)
            source_info["git_committer_email"] = getattr(commit.committer, "email", None)

            if repo.working_tree_dir:
                source_info["relative_path"] = os.path.relpath(source_info["file_path"], repo.working_tree_dir)

            try:
                source_info["git_branch"] = repo.active_branch.name
            except Exception:
                source_info["git_branch"] = None

            matching_tags = [tag.name for tag in repo.tags if tag.commit == repo.head.commit]
            if matching_tags:
                source_info["git_tag"] = matching_tags[0]

            try:
                source_info["repo_url"] = next(iter(repo.remotes.origin.urls), None)
                source_info["repo_owner"] = self._extract_repo_owner(source_info["repo_url"])
            except Exception:
                source_info["repo_url"] = None
                source_info["repo_owner"] = None
        except Exception:
            pass

        self._op_source_cache[op_class] = source_info
        return dict(source_info)

    def _resolve_config_path(self) -> str | None:
        config = ConfigAccessor.get(self.cfg, "config", None)
        if isinstance(config, list) and config:
            return str(config[0])
        if config is None:
            return None
        return str(config)

    def _build_pipeline_recipe(self) -> Any:
        return self._to_jsonable(self.cfg)

    @staticmethod
    def _hash_op_config(op_config: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(op_config, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_recipe(recipe: Any) -> str:
        return hashlib.sha256(json.dumps(recipe, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    @classmethod
    def _to_jsonable(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): cls._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._to_jsonable(v) for v in value]
        if hasattr(value, "__dict__"):
            return {str(k): cls._to_jsonable(v) for k, v in vars(value).items() if not str(k).startswith("_")}
        return str(value)

    @staticmethod
    def _extract_repo_owner(repo_url: str | None) -> str | None:
        if not repo_url:
            return None

        normalized = repo_url.rstrip("/")
        if normalized.endswith(".git"):
            normalized = normalized[:-4]

        if "://" in normalized:
            path = normalized.split("://", 1)[1]
            if "/" in path:
                path = path.split("/", 1)[1]
            parts = [part for part in path.split("/") if part]
            if len(parts) >= 2:
                return parts[-2]
            return None

        if ":" in normalized:
            path = normalized.split(":", 1)[1]
            parts = [part for part in path.split("/") if part]
            if len(parts) >= 2:
                return parts[-2]
        return None

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
