#!/usr/bin/env bash
# ONLINE step: download a model profile's weights, the Open WebUI embedding
# model, and the Docker images. Usage: ./scripts/prepare.sh <profile> [--export-images]
set -euo pipefail
cd "$(dirname "$0")/.."

PROFILE="${1:-}"
if [[ -z "$PROFILE" || ! -f "profiles/$PROFILE.env" ]]; then
  echo "Usage: $0 <profile> [--export-images]"; echo "Profiles:"
  ls profiles | sed 's/\.env$//; s/^/  /'; exit 1
fi
set -a; source config.env; source "profiles/$PROFILE.env"; set +a

mkdir -p models/_embeddings
echo ">> Downloading $HF_REPO -> models/$MODEL_DIR (and embedding model)"
docker run --rm -v "$PWD/models:/models" -e HF_TOKEN="${HF_TOKEN:-}" python:3.12-slim sh -c "
  pip install -q -U huggingface_hub &&
  hf download '$HF_REPO' --local-dir '/models/$MODEL_DIR' &&
  hf download sentence-transformers/all-MiniLM-L6-v2 --local-dir /models/_embeddings/all-MiniLM-L6-v2 &&
  chown -R $(id -u):$(id -g) /models"

echo ">> Pulling images (vllm/vllm-openai:$VLLM_TAG, open-webui:$OPENWEBUI_TAG)"
docker compose --env-file config.env --env-file "profiles/$PROFILE.env" pull

if [[ "${2:-}" == "--export-images" ]]; then
  OUT="images-$PROFILE.tar"
  echo ">> Saving images to $OUT (on the offline host: docker load -i $OUT)"
  docker save -o "$OUT" "vllm/vllm-openai:$VLLM_TAG" "ghcr.io/open-webui/open-webui:$OPENWEBUI_TAG"
fi
echo ">> Done. Start with: ./scripts/run.sh $PROFILE"
