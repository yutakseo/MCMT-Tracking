#!/bin/bash
set -euo pipefail

# ====== 설정 ======
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"          # ← 프로젝트 루트(여기에 requirements.txt가 있음)

DOCKER_IMAGE="aei_with_tracking"
CONTAINER_NAME="MCMT-container"
DOCKERFILE_PATH="${SCRIPT_DIR}/Dockerfile"
DATASETS_FILE="${SCRIPT_DIR}/___DATASETS___.list"

DEBUG=true  # 디버깅 출력 활성화 여부

# 기본값
VOLUME_DIR="$(pwd)"
VOLUME_FLAGS=""

print_debug() { $DEBUG && echo -e "[DEBUG] $1"; }

parse_args() {
  while [[ "${1:-}" != "" ]]; do
    case "$1" in
      -v|--volume) VOLUME_DIR="$2"; shift ;;
      *) echo "[ERROR] Unknown parameter: $1"; exit 1 ;;
    esac
    shift
  done
  VOLUME_DIR="$(realpath "$VOLUME_DIR")"
  VOLUME_FLAGS="-v ${VOLUME_DIR}:/workspace"
  print_debug "VOLUME_DIR=${VOLUME_DIR}"
}

build_image_if_needed() {
  if ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
    echo "[INFO] Building Docker image: $DOCKER_IMAGE"
    # ★ 컨텍스트를 프로젝트 루트로 변경
    docker build -t "$DOCKER_IMAGE" -f "$DOCKERFILE_PATH" "$PROJECT_ROOT"
    echo "[INFO] Docker image built: $DOCKER_IMAGE"
  else
    echo "[INFO] Docker image already exists: $DOCKER_IMAGE"
  fi
}

remove_existing_container() {
  if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    docker rm -f "$CONTAINER_NAME" >/dev/null
    echo "[INFO] Removed existing container: $CONTAINER_NAME"
  fi
}

prepare_volume_mounts() {
  if [[ -f "$DATASETS_FILE" ]]; then
    mapfile -t dataset_paths < "$DATASETS_FILE"
    for dataset_path in "${dataset_paths[@]}"; do
      [[ -z "$dataset_path" || "$dataset_path" =~ ^# ]] && continue
      dataset_path_clean="$(realpath "$dataset_path" 2>/dev/null)" || { echo "[WARN] Invalid path: $dataset_path"; continue; }
      dataset_name="$(basename "$dataset_path_clean")"
      [[ -z "$dataset_name" || "$dataset_name" == "." ]] && { echo "[WARN] Skipping invalid dataset name: $dataset_path_clean"; continue; }
      VOLUME_FLAGS+=" -v ${dataset_path_clean}:/workspace/datasets/${dataset_name}/"
    done
    echo "[INFO] Volume mount list parsed from $DATASETS_FILE"
  else
    echo "[INFO] No dataset list file found. Skipping additional mounts."
  fi
  print_debug "Volume flags: $VOLUME_FLAGS"
}

run_container() {
  echo "[INFO] Starting container: $CONTAINER_NAME"
  docker run -dit --gpus all \
    $VOLUME_FLAGS \
    --shm-size=64g \
    --restart unless-stopped \
    --name "$CONTAINER_NAME" \
    "$DOCKER_IMAGE" \
    /bin/bash
  echo "[INFO] Container started in background: $CONTAINER_NAME"
}

init_container_filesystem() {
  # 컨테이너가 확실히 떠있는지 체크
  if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[ERROR] Container did not start properly. Aborting init."
    exit 1
  fi
  docker exec "$CONTAINER_NAME" mkdir -p /workspace/datasets
  echo "[INFO] Created /workspace/datasets in container"

  # /opt/requirements.txt 는 Dockerfile에서 COPY한 파일
  docker exec "$CONTAINER_NAME" ln -sf /opt/requirements.txt /workspace/requirements.txt
  echo "[INFO] Created symbolic link for requirements.txt"
}

echo "[INFO] Starting Docker container setup..."
parse_args "$@"
echo "[INFO] Parsed input arguments."

build_image_if_needed
remove_existing_container
prepare_volume_mounts
run_container
init_container_filesystem

echo "[INFO] Container is up and ready: $CONTAINER_NAME 🎉"
