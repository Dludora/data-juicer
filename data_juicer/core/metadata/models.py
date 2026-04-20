from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OpKey:
    partition_id: int
    op_index: int


@dataclass
class SchemaField:
    name: str
    type: str | None = None
    fields: list["SchemaField"] = field(default_factory=list)
    nullable: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetRef:
    namespace: str
    name: str
    role: str
    source_type: str
    uri: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetSnapshot:
    refs: list[DatasetRef] = field(default_factory=list)
    rows: int | None = None
    schema: list[SchemaField] = field(default_factory=list)
    storage_kind: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class OpCtx:
    op_id: str
    op_name: str
    op_type: str
    op_index: int
    partition_id: int = 0
    started_at: str | None = None
    ended_at: str | None = None
    status: str = "started"
    input_snapshot: DatasetSnapshot | None = None
    output_snapshot: DatasetSnapshot | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Ctx:
    run_id: str
    job_id: str | None
    executor_type: str
    job_name: str
    namespace: str
    project_name: str
    started_at: str
    ended_at: str | None = None
    status: str = "started"
    input_snapshot: DatasetSnapshot | None = None
    output_snapshot: DatasetSnapshot | None = None
    op_ctxs: dict[OpKey, OpCtx] = field(default_factory=dict)
    latest_snapshot_by_partition: dict[int, DatasetSnapshot] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
