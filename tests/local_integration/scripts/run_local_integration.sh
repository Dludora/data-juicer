#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
PREPARE_SCRIPT="$ROOT_DIR/tests/local_integration/scripts/prepare_local_integration_assets.py"
PROCESS_DATA_SCRIPT="$ROOT_DIR/tools/process_data.py"
COMPOSE_FILE="$ROOT_DIR/tests/local_integration/hdfs/docker-compose.yml"
RECIPES_DIR="$ROOT_DIR/tests/local_integration/runtime/recipes"
RUNTIME_HDFS_DIR="$ROOT_DIR/tests/local_integration/runtime/hdfs"
HDFS_SEED_DIR="$ROOT_DIR/tests/local_integration/runtime/hdfs/seed"
NAMENODE_CONTAINER="dj-hdfs-namenode"

compose() {
  if docker compose version >/dev/null 2>&1; then
    docker compose -f "$COMPOSE_FILE" "$@"
  else
    docker-compose -f "$COMPOSE_FILE" "$@"
  fi
}

prepare() {
  "$PYTHON_BIN" "$PREPARE_SCRIPT"
}

run_recipe() {
  local recipe_path="$1"
  "$PYTHON_BIN" "$PROCESS_DATA_SCRIPT" --config "$recipe_path"
}

wait_for_hdfs() {
  local retries=30
  local sleep_secs=2

  for ((i=1; i<=retries; i++)); do
    if compose exec -T namenode hdfs dfs -ls / >/dev/null 2>&1; then
      return 0
    fi
    sleep "$sleep_secs"
  done

  echo "HDFS did not become ready in time." >&2
  return 1
}

hdfs_up() {
  prepare
  mkdir -p "$RUNTIME_HDFS_DIR/namenode" "$RUNTIME_HDFS_DIR/datanode"
  compose up -d
  wait_for_hdfs
}

hdfs_down() {
  compose down
}

hdfs_seed() {
  hdfs_up
  compose exec -T namenode hdfs dfs -rm -r -f /data-juicer >/dev/null 2>&1 || true
  compose exec -T namenode hdfs dfs -mkdir -p /data-juicer
  docker exec "$NAMENODE_CONTAINER" sh -c 'rm -rf /tmp/data-juicer-seed && mkdir -p /tmp/data-juicer-seed'
  docker cp "$HDFS_SEED_DIR/." "$NAMENODE_CONTAINER:/tmp/data-juicer-seed/"
  compose exec -T namenode sh -c '/opt/hadoop-3.2.1/bin/hdfs dfs -put -f /tmp/data-juicer-seed/* /data-juicer/'
  compose exec -T namenode hdfs dfs -ls /data-juicer
}

hdfs_all() {
  hdfs_seed
  run_recipe "$RECIPES_DIR/hdfs/default_load_jsonl.yaml"
  run_recipe "$RECIPES_DIR/hdfs/default_load_json.yaml"
  run_recipe "$RECIPES_DIR/hdfs/default_load_csv.yaml"
  run_recipe "$RECIPES_DIR/hdfs/default_load_tsv.yaml"
  run_recipe "$RECIPES_DIR/hdfs/default_load_txt.yaml"
  run_recipe "$RECIPES_DIR/hdfs/default_load_parquet.yaml"
}

iceberg_all() {
  prepare
  run_recipe "$RECIPES_DIR/iceberg/default_export_to_iceberg.yaml"
  run_recipe "$RECIPES_DIR/iceberg/default_load_from_iceberg.yaml"
  run_recipe "$RECIPES_DIR/iceberg/ray_load_from_iceberg.yaml"
}

paimon_all() {
  prepare
  run_recipe "$RECIPES_DIR/paimon/default_export_to_paimon.yaml"
  run_recipe "$RECIPES_DIR/paimon/ray_load_from_paimon.yaml"
}

run_all() {
  prepare
  hdfs_all
  run_recipe "$RECIPES_DIR/iceberg/default_export_to_iceberg.yaml"
  run_recipe "$RECIPES_DIR/iceberg/default_load_from_iceberg.yaml"
  run_recipe "$RECIPES_DIR/iceberg/ray_load_from_iceberg.yaml"
  run_recipe "$RECIPES_DIR/paimon/default_export_to_paimon.yaml"
  run_recipe "$RECIPES_DIR/paimon/ray_load_from_paimon.yaml"
}

usage() {
  cat <<'EOF'
Usage:
  bash tests/local_integration/scripts/run_local_integration.sh prepare
  bash tests/local_integration/scripts/run_local_integration.sh hdfs-up
  bash tests/local_integration/scripts/run_local_integration.sh hdfs-seed
  bash tests/local_integration/scripts/run_local_integration.sh hdfs
  bash tests/local_integration/scripts/run_local_integration.sh iceberg
  bash tests/local_integration/scripts/run_local_integration.sh paimon
  bash tests/local_integration/scripts/run_local_integration.sh all
  bash tests/local_integration/scripts/run_local_integration.sh hdfs-down
EOF
}

main() {
  local target="${1:-}"

  case "$target" in
    prepare)
      prepare
      ;;
    hdfs-up)
      hdfs_up
      ;;
    hdfs-seed)
      hdfs_seed
      ;;
    hdfs)
      hdfs_all
      ;;
    iceberg)
      iceberg_all
      ;;
    paimon)
      paimon_all
      ;;
    all)
      run_all
      ;;
    hdfs-down)
      hdfs_down
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
