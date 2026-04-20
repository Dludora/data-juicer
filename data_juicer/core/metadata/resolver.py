from __future__ import annotations

import os
from typing import Any, List, Tuple, get_args, get_origin
from urllib.parse import urlparse

from data_juicer.core.data.schema import Schema
from data_juicer.core.metadata.models import DatasetRef, DatasetSnapshot, SchemaField
from data_juicer.utils.config_utils import ConfigAccessor


def _parse_weighted_dataset_path(dataset_path: str) -> list[str]:
    if not dataset_path:
        return []

    tokens = dataset_path.split()
    paths: list[str] = []
    for token in tokens:
        try:
            value = float(token)
            if 0.0 <= value <= 1.0:
                continue
        except ValueError:
            pass
        paths.append(token)
    return paths


def _resolve_file_like(uri: str) -> Tuple[str, str, str]:
    parsed = urlparse(uri)
    if parsed.scheme == "s3":
        namespace = f"s3://{parsed.netloc}"
        name = parsed.path.lstrip("/")
        return namespace, name, "s3"

    if parsed.scheme == "hdfs":
        host = parsed.hostname or parsed.netloc or ""
        port = parsed.port or 8020
        namespace = f"hdfs://{host}:{port}"
        name = parsed.path
        return namespace, name, "hdfs"

    if parsed.scheme in {"file", "gs", "abfss", "wasbs", "dbfs"}:
        namespace = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else parsed.scheme
        name = parsed.path.lstrip("/") if parsed.path else parsed.netloc
        return namespace, name or "unknown", parsed.scheme

    if parsed.scheme:
        namespace = f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else parsed.scheme
        name = parsed.path.lstrip("/") or parsed.netloc or "unknown"
        return namespace, name, parsed.scheme

    abs_path = os.path.abspath(uri)
    return "file", abs_path, "local"


def _resolve_iceberg_table(table_identifier: str, catalog_kwargs: Any) -> Tuple[str, str, dict[str, Any]]:
    catalog_name = ConfigAccessor.get(catalog_kwargs, "name", None)
    catalog_type = ConfigAccessor.get(catalog_kwargs, "type", None)
    catalog_uri = ConfigAccessor.get(catalog_kwargs, "uri", None)
    warehouse = ConfigAccessor.get(catalog_kwargs, "warehouse", None)

    namespace_token = catalog_name or catalog_uri or "default"
    namespace = f"iceberg://{namespace_token}"
    extra = {
        "catalog_name": catalog_name,
        "catalog_type": catalog_type,
        "catalog_uri": catalog_uri,
        "warehouse": warehouse,
    }
    return namespace, table_identifier, extra


class MetadataResolver:
    def __init__(self, cfg: Any):
        self.cfg = cfg
        metadata_cfg = ConfigAccessor.get(cfg, "metadata", {})
        capture_cfg = ConfigAccessor.get(metadata_cfg, "capture", {})
        self.capture_schema = bool(ConfigAccessor.get(capture_cfg, "schema", True))
        self.capture_rows = bool(ConfigAccessor.get(capture_cfg, "rows", False))

    def resolve_input_refs(self) -> list[DatasetRef]:
        refs: list[DatasetRef] = []

        dataset_path = ConfigAccessor.get(self.cfg, "dataset_path", "")
        for uri in _parse_weighted_dataset_path(dataset_path):
            namespace, name, source_type = _resolve_file_like(uri)
            refs.append(
                DatasetRef(
                    namespace=namespace,
                    name=name,
                    role="input",
                    source_type=source_type,
                    uri=uri,
                )
            )

        dataset_cfg = ConfigAccessor.get(self.cfg, "dataset", None)
        for item in self._normalize_dataset_configs(dataset_cfg):
            source = ConfigAccessor.get(item, "source", None)
            item_type = ConfigAccessor.get(item, "type", None)

            if source == "iceberg":
                table_identifier = ConfigAccessor.get(item, "table_identifier", None)
                if table_identifier:
                    namespace, name, extra = _resolve_iceberg_table(
                        table_identifier, ConfigAccessor.get(item, "catalog_kwargs", {})
                    )
                    refs.append(
                        DatasetRef(
                            namespace=namespace,
                            name=name,
                            role="input",
                            source_type="iceberg",
                            uri=table_identifier,
                            extra=extra,
                        )
                    )
                continue

            path = ConfigAccessor.get(item, "path", None)
            if path:
                namespace, name, source_type = _resolve_file_like(path)
                resolved_source_type = source or ("remote" if item_type == "remote" else source_type)
                refs.append(
                    DatasetRef(
                        namespace=namespace,
                        name=name,
                        role="input",
                        source_type=resolved_source_type,
                        uri=path,
                        extra={
                            "endpoint_url": ConfigAccessor.get(item, "endpoint_url", None),
                            "host": ConfigAccessor.get(item, "host", None),
                            "port": ConfigAccessor.get(item, "port", None),
                        },
                    )
                )

        return self._dedupe_refs(refs)

    def resolve_output_refs(self) -> list[DatasetRef]:
        refs: list[DatasetRef] = []

        export_type = ConfigAccessor.get(self.cfg, "export_type", None)
        export_path = ConfigAccessor.get(self.cfg, "export_path", None)

        if export_type == "iceberg":
            export_extra = ConfigAccessor.get(self.cfg, "export_extra_args", {})
            table_identifier = ConfigAccessor.get(export_extra, "table_identifier", None)
            catalog_kwargs = ConfigAccessor.get(export_extra, "catalog_kwargs", {})
            if table_identifier:
                namespace, name, extra = _resolve_iceberg_table(table_identifier, catalog_kwargs)
                refs.append(
                    DatasetRef(
                        namespace=namespace,
                        name=name,
                        role="output",
                        source_type="iceberg",
                        uri=table_identifier,
                        extra=extra,
                    )
                )
                return refs

        if export_path:
            namespace, name, source_type = _resolve_file_like(export_path)
            refs.append(
                DatasetRef(
                    namespace=namespace,
                    name=name,
                    role="output",
                    source_type=source_type,
                    uri=export_path,
                )
            )

        return refs

    def build_inmemory_ref(self, run_id: str, node_name: str, role: str) -> DatasetRef:
        return DatasetRef(
            namespace=f"inmemory://{run_id}",
            name=node_name,
            role=role,
            source_type="inmemory",
            uri=f"inmemory://{run_id}/{node_name}",
        )

    def snapshot(
        self,
        dataset_obj: Any | None,
        refs: list[DatasetRef] | None = None,
        storage_kind: str | None = None,
    ) -> DatasetSnapshot | None:
        if dataset_obj is None and not refs:
            return None

        schema = self._extract_schema_fields(dataset_obj) if self.capture_schema else []
        resolved_storage_kind = storage_kind or self._resolve_storage_kind(dataset_obj, refs or [])
        rows = self._extract_rows(dataset_obj) if self.capture_rows else None
        return DatasetSnapshot(
            refs=refs or [],
            rows=rows,
            schema=schema,
            storage_kind=resolved_storage_kind,
            extra={},
        )

    def build_pipeline_input_snapshot(self, dataset_obj: Any | None) -> DatasetSnapshot | None:
        refs = self.resolve_input_refs()
        return self.snapshot(dataset_obj=dataset_obj, refs=refs)

    def build_pipeline_output_snapshot(self, dataset_obj: Any | None) -> DatasetSnapshot | None:
        refs = self.resolve_output_refs()
        return self.snapshot(dataset_obj=dataset_obj, refs=refs)

    def build_inmemory_snapshot(
        self,
        run_id: str,
        node_name: str,
        dataset_obj: Any | None,
        role: str,
    ) -> DatasetSnapshot:
        ref = self.build_inmemory_ref(run_id, node_name, role)
        snapshot = self.snapshot(dataset_obj=dataset_obj, refs=[ref], storage_kind="inmemory")
        return snapshot or DatasetSnapshot(refs=[ref], storage_kind="inmemory")

    def _extract_schema_fields(self, dataset_obj: Any | None) -> list[SchemaField]:
        if dataset_obj is None:
            return []

        schema = None
        if hasattr(dataset_obj, "schema") and callable(getattr(dataset_obj, "schema")):
            try:
                schema = dataset_obj.schema()
            except Exception:
                schema = None

        if schema is None and hasattr(dataset_obj, "features"):
            try:
                schema = Schema.from_hf_features(dataset_obj.features)
            except Exception:
                schema = None

        if schema is None:
            return []

        return self._schema_to_fields(schema)

    def _schema_to_fields(self, schema: Any) -> list[SchemaField]:
        fields: list[SchemaField] = []
        for col in getattr(schema, "columns", []) or []:
            field_type = getattr(schema, "column_types", {}).get(col)
            fields.append(self._build_schema_field(col, field_type))
        return fields

    def _build_schema_field(self, field_name: str, field_type: Any) -> SchemaField:
        if isinstance(field_type, Schema):
            return SchemaField(
                name=field_name,
                type="struct",
                fields=self._schema_to_fields(field_type),
            )

        origin = get_origin(field_type)
        args = get_args(field_type)
        if origin in {list, List}:
            extra: dict[str, Any] = {}
            if args:
                extra["item_type"] = self._python_type_name(args[0])
            return SchemaField(name=field_name, type="array", extra=extra)

        return SchemaField(name=field_name, type=self._python_type_name(field_type))

    def _python_type_name(self, value: Any) -> str:
        if isinstance(value, Schema):
            return "struct"

        origin = get_origin(value)
        args = get_args(value)
        if origin in {list, List}:
            if args:
                return f"array[{self._python_type_name(args[0])}]"
            return "array"

        if value is None:
            return "unknown"

        if isinstance(value, type):
            return value.__name__

        return str(value).replace("typing.", "")

    def _extract_rows(self, dataset_obj: Any | None) -> int | None:
        if dataset_obj is None:
            return None

        try:
            if hasattr(dataset_obj, "count") and callable(getattr(dataset_obj, "count")):
                return int(dataset_obj.count())
        except Exception:
            pass

        try:
            return len(dataset_obj)  # type: ignore[arg-type]
        except Exception:
            return None

    def _resolve_storage_kind(self, dataset_obj: Any | None, refs: list[DatasetRef]) -> str | None:
        if dataset_obj is not None:
            class_name = dataset_obj.__class__.__name__.lower()
            if "raydataset" in class_name:
                return "ray"
            if "dataset" in class_name:
                return "hf"

        if refs:
            source_type = refs[0].source_type
            if source_type in {"s3", "hdfs", "local", "file", "iceberg", "inmemory"}:
                return source_type
            return source_type

        return None

    @staticmethod
    def _normalize_dataset_configs(dataset_cfg: Any) -> list[Any]:
        if not dataset_cfg:
            return []
        if isinstance(dataset_cfg, list):
            return dataset_cfg
        if isinstance(dataset_cfg, dict):
            return ConfigAccessor.get(dataset_cfg, "configs", []) or [dataset_cfg]
        nested = ConfigAccessor.get(dataset_cfg, "configs", [])
        return nested or [dataset_cfg]

    @staticmethod
    def _dedupe_refs(refs: list[DatasetRef]) -> list[DatasetRef]:
        deduped: dict[str, DatasetRef] = {}
        for ref in refs:
            key = f"{ref.role}:{ref.namespace}:{ref.name}"
            deduped[key] = ref
        return list(deduped.values())
