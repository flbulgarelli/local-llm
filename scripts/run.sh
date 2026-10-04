#!/usr/bin/env bash
# OFFLINE step: start (or switch to) a model profile.
# Usage: ./scripts/run.sh <profile> | stop | logs | status
set -euo pipefail
cd "$(dirname "$0")/.."

compose() { docker compose --env-file config.env --env-file "profiles/$1.env" "${@:2}"; }
active() { [[ -f .active-profile ]] && cat .active-profile || { echo "No active profile" >&2; exit 1; }; }

case "${1:-}" in
  "" )
    echo "Usage: $0 <profile> | stop | logs | status"; echo "Profiles:"
    ls profiles | sed 's/\.env$//; s/^/  /'; exit 1 ;;
  stop )   compose "$(active)" down ;;
  logs )   compose "$(active)" logs -f vllm ;;
  status ) echo "Active profile: $(active)"; compose "$(active)" ps ;;
  * )
    PROFILE="$1"
    [[ -f "profiles/$PROFILE.env" ]] || { echo "Unknown profile: $PROFILE"; exit 1; }
    set -a; source config.env; source "profiles/$PROFILE.env"; set +a
    [[ "${MODEL_RUNTIME:-runc}" == "nvidia" ]] && export VLLM_IMAGE="vllm/vllm-openai" || export VLLM_IMAGE="vllm/vllm-openai-cpu"
    [[ -f "models/$MODEL_DIR/config.json" ]] || { echo "Model missing in models/$MODEL_DIR. Run ./scripts/prepare.sh $PROFILE (online) first."; exit 1; }
    echo "$PROFILE" > .active-profile
    # "up -d" recreates only vllm when the profile changed; Open WebUI (and its chats) stays.
    compose "$PROFILE" up -d
    echo "Starting $SERVED_MODEL_NAME. Loading can take a few minutes: ./scripts/run.sh logs"
    echo "Open WebUI: http://localhost:${WEBUI_PORT:-3000}" ;;
esac
