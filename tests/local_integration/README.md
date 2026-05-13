# Local Integration Tests

This directory is for local, environment-backed validation of remote/table data flows. These are intentionally not part of the regular unittest suite.

## What is covered

- `HDFS`
  - `DefaultHDFSDataLoadStrategy`
  - Real HDFS container via Docker
  - Real file formats: `jsonl`, `json`, `csv`, `tsv`, `txt`, `parquet`
  - Validation path: `tools/process_data.py --config <recipe>`

- `Iceberg`
  - Default-mode export to Iceberg
  - Default-mode load from Iceberg
  - Ray-mode load from Iceberg
  - Local SQL catalog + local warehouse

- `Paimon`
  - Default-mode export to Paimon
  - Ray-mode load from Paimon
  - Local filesystem catalog + local warehouse

## Prerequisites

Install the distributed dependencies first:

```bash
uv sync --extra distributed --extra dev
```

If your local environment was created before these integration dependencies were pinned, rerun the sync command so `pyarrow`, `pyiceberg`, `pypaimon`, and `ray` match the expected local test setup.

For HDFS, make sure Docker is available:

```bash
docker version
docker compose version
```

## Quick Start

From the repository root:

```bash
bash tests/local_integration/scripts/run_local_integration.sh prepare
```

This generates:

- local source files
- local HDFS seed files
- local Paimon warehouse and empty table
- ready-to-run recipe YAML files under `tests/local_integration/runtime/recipes`

## Run Targets

Run all HDFS recipes:

```bash
bash tests/local_integration/scripts/run_local_integration.sh hdfs
```

Run the full Iceberg flow:

```bash
bash tests/local_integration/scripts/run_local_integration.sh iceberg
```

Run the full Paimon flow:

```bash
bash tests/local_integration/scripts/run_local_integration.sh paimon
```

Run everything:

```bash
bash tests/local_integration/scripts/run_local_integration.sh all
```

Stop the local HDFS cluster:

```bash
bash tests/local_integration/scripts/run_local_integration.sh hdfs-down
```

## Notes

- Generated recipes live under `tests/local_integration/runtime/recipes`.
- Generated outputs live under `tests/local_integration/runtime/outputs`.
- HDFS uses a local Docker cluster exposed at `hdfs://localhost:9000`.
- Paimon currently uses `ray` for load because there is no default-mode Paimon load strategy in the codebase.
- Iceberg and Paimon local assets are recreated during `prepare`, so rerunning starts from a clean local table state.
