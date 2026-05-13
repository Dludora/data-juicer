import csv
import getpass
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_INTEGRATION_DIR = REPO_ROOT / "tests" / "local_integration"
RUNTIME_DIR = LOCAL_INTEGRATION_DIR / "runtime"
RECIPES_DIR = RUNTIME_DIR / "recipes"
OUTPUTS_DIR = RUNTIME_DIR / "outputs"
SOURCES_DIR = RUNTIME_DIR / "sources"
HDFS_SEED_DIR = RUNTIME_DIR / "hdfs" / "seed"
ICEBERG_WAREHOUSE_DIR = RUNTIME_DIR / "iceberg_warehouse"
ICEBERG_CATALOG_DB = RUNTIME_DIR / "iceberg_catalog.db"
PAIMON_WAREHOUSE_DIR = RUNTIME_DIR / "paimon_warehouse"


RECORDS = [
    {
        "text": "alpha sample for local integration testing",
        "doc_id": 1,
        "lang": "en",
        "score": 0.98,
        "is_valid": True,
    },
    {
        "text": "beta sample used for exporter and loader verification",
        "doc_id": 2,
        "lang": "en",
        "score": 0.87,
        "is_valid": False,
    },
    {
        "text": "gamma sample keeps the recipe path realistic",
        "doc_id": 3,
        "lang": "fr",
        "score": 0.91,
        "is_valid": True,
    },
]


def _reset_runtime_dirs():
    for path in [RECIPES_DIR, OUTPUTS_DIR, SOURCES_DIR, HDFS_SEED_DIR]:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    if ICEBERG_WAREHOUSE_DIR.exists():
        shutil.rmtree(ICEBERG_WAREHOUSE_DIR)
    ICEBERG_WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)

    if PAIMON_WAREHOUSE_DIR.exists():
        shutil.rmtree(PAIMON_WAREHOUSE_DIR)
    PAIMON_WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)

    if ICEBERG_CATALOG_DB.exists():
        ICEBERG_CATALOG_DB.unlink()


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def _write_csv_like(path: Path, rows, delimiter: str):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["text", "doc_id", "lang", "score", "is_valid"],
            delimiter=delimiter,
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_txt(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row["text"] + "\n")


def _write_parquet(path: Path, rows):
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _prepare_source_files():
    _write_jsonl(SOURCES_DIR / "integration_documents.jsonl", RECORDS)
    _write_parquet(SOURCES_DIR / "integration_documents.parquet", RECORDS)

    _write_jsonl(HDFS_SEED_DIR / "sample.jsonl", RECORDS)
    _write_json(HDFS_SEED_DIR / "sample.json", RECORDS)
    _write_csv_like(HDFS_SEED_DIR / "sample.csv", RECORDS, ",")
    _write_csv_like(HDFS_SEED_DIR / "sample.tsv", RECORDS, "\t")
    _write_txt(HDFS_SEED_DIR / "sample.txt", RECORDS)
    _write_parquet(HDFS_SEED_DIR / "sample.parquet", RECORDS)


def _prepare_paimon_table():
    try:
        from pypaimon import Schema
        from pypaimon.catalog.catalog_factory import CatalogFactory
    except ImportError as e:
        raise RuntimeError(
            "Preparing local Paimon assets requires pypaimon. "
            "Install the distributed extra before running local integration tests."
        ) from e

    table = pa.Table.from_pylist(RECORDS)
    catalog_options = {
        "warehouse": f"file://{PAIMON_WAREHOUSE_DIR}",
    }
    catalog = CatalogFactory.create(dict(catalog_options))
    catalog.create_database("default", ignore_if_exists=True)

    if hasattr(Schema, "from_pyarrow_schema"):
        schema = Schema.from_pyarrow_schema(pa_schema=table.schema)
    else:
        schema = Schema(pa_schema=table.schema)

    catalog.create_table(
        "default.integration_documents",
        schema=schema,
        ignore_if_exists=False,
    )


def _dump_yaml(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def _base_recipe(export_path: str):
    return {
        "project_name": "local-integration",
        "np": 1,
        "text_keys": "text",
        "export_path": export_path,
        "process": [],
    }


def _render_hdfs_recipes():
    user_name = getpass.getuser()
    formats = ["jsonl", "json", "csv", "tsv", "txt", "parquet"]

    for file_format in formats:
        payload = _base_recipe(str(OUTPUTS_DIR / "hdfs" / f"{file_format}.jsonl"))
        payload["dataset"] = {
            "configs": [
                {
                    "type": "remote",
                    "source": "hdfs",
                    "path": f"hdfs://localhost:9000/data-juicer/sample.{file_format}",
                    "host": "localhost",
                    "port": 9000,
                    "user": user_name,
                }
            ]
        }
        _dump_yaml(RECIPES_DIR / "hdfs" / f"default_load_{file_format}.yaml", payload)


def _iceberg_catalog_kwargs():
    return {
        "name": "local",
        "type": "sql",
        "uri": f"sqlite:///{ICEBERG_CATALOG_DB}",
        "warehouse": f"file://{ICEBERG_WAREHOUSE_DIR}",
    }


def _render_iceberg_recipes():
    export_recipe = _base_recipe(str(OUTPUTS_DIR / "iceberg" / "default_export_placeholder.jsonl"))
    export_recipe["dataset"] = {
        "configs": [
            {
                "type": "local",
                "path": str(SOURCES_DIR / "integration_documents.parquet"),
            }
        ]
    }
    export_recipe["export_type"] = "iceberg"
    export_recipe["export_extra_args"] = {
        "table_identifier": "default.integration_documents",
        "catalog_kwargs": _iceberg_catalog_kwargs(),
    }
    _dump_yaml(RECIPES_DIR / "iceberg" / "default_export_to_iceberg.yaml", export_recipe)

    default_load_recipe = _base_recipe(str(OUTPUTS_DIR / "iceberg" / "default_load_output.jsonl"))
    default_load_recipe["dataset"] = {
        "configs": [
            {
                "type": "remote",
                "source": "iceberg",
                "table_identifier": "default.integration_documents",
                "catalog_kwargs": _iceberg_catalog_kwargs(),
            }
        ]
    }
    _dump_yaml(RECIPES_DIR / "iceberg" / "default_load_from_iceberg.yaml", default_load_recipe)

    ray_load_recipe = _base_recipe(str(OUTPUTS_DIR / "iceberg" / "ray_load_output"))
    ray_load_recipe["executor_type"] = "ray"
    ray_load_recipe["ray_address"] = "local"
    ray_load_recipe["export_type"] = "jsonl"
    ray_load_recipe["dataset"] = default_load_recipe["dataset"]
    _dump_yaml(RECIPES_DIR / "iceberg" / "ray_load_from_iceberg.yaml", ray_load_recipe)


def _paimon_catalog_options():
    return {
        "warehouse": f"file://{PAIMON_WAREHOUSE_DIR}",
    }


def _render_paimon_recipes():
    export_recipe = _base_recipe(str(OUTPUTS_DIR / "paimon" / "default_export_placeholder.jsonl"))
    export_recipe["dataset"] = {
        "configs": [
            {
                "type": "local",
                "path": str(SOURCES_DIR / "integration_documents.parquet"),
            }
        ]
    }
    export_recipe["export_type"] = "paimon"
    export_recipe["export_extra_args"] = {
        "table_identifier": "default.integration_documents",
        "catalog_options": _paimon_catalog_options(),
    }
    _dump_yaml(RECIPES_DIR / "paimon" / "default_export_to_paimon.yaml", export_recipe)

    ray_load_recipe = _base_recipe(str(OUTPUTS_DIR / "paimon" / "ray_load_output"))
    ray_load_recipe["executor_type"] = "ray"
    ray_load_recipe["ray_address"] = "local"
    ray_load_recipe["export_type"] = "jsonl"
    ray_load_recipe["dataset"] = {
        "configs": [
            {
                "type": "remote",
                "source": "paimon",
                "table_identifier": "default.integration_documents",
                "catalog_options": _paimon_catalog_options(),
            }
        ]
    }
    _dump_yaml(RECIPES_DIR / "paimon" / "ray_load_from_paimon.yaml", ray_load_recipe)


def main():
    _reset_runtime_dirs()
    _prepare_source_files()
    _prepare_paimon_table()
    _render_hdfs_recipes()
    _render_iceberg_recipes()
    _render_paimon_recipes()

    print("Prepared local integration assets:")
    print(f"  Recipes: {RECIPES_DIR}")
    print(f"  Sources: {SOURCES_DIR}")
    print(f"  HDFS seed files: {HDFS_SEED_DIR}")


if __name__ == "__main__":
    main()
