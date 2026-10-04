# Local LLM

> Offline GPU LLM stack

Este proyecto contiene scrips de instalación y configuración de un entorno de
ejecución LLM open source.


## Arquitectura


Tiene soporte para varios modelos, pero sólo úno funciona por vez, para lo cual se definen múltiples archivos de perfiles y luego se inician haciendo:

```bash
./scripts/run.sh <profile>
```

Esto levanta al modelo utilizando `vLLM` y exponiendo un API estilo OpenAI en el puerto 8000. Además, expone una interfaz Web en el puerto 3000 mediante `Open WebUI`

```
Browser ──► Open WebUI (:3000) ──► vLLM (:8000, compatible con OpenAI API) ──► ./models/<model>
```

## Project layout

```
local-llm/
├── docs/HARDWARE.md        RAM / VRAM / CPU / card requirements per model
├── docker-compose.yml      vLLM + Open WebUI, fully parameterized (no need to edit)
├── config.env              shared settings: API key, ports, Open WebUI version
├── profiles/               one .env per model (repo, folder, vLLM version, flags)
├── scripts/prepare.sh      ONLINE: download weights + embedding model, pull images
├── scripts/run.sh          OFFLINE: start / switch / stop / logs / status
└── models/                 weights (filled by prepare.sh), mounted read-only
```

## Profiles

| Profile                 | Model                        | Min VRAM | Notes                                |
|-------------------------|------------------------------|----------|--------------------------------------|
| `qwen3.8-27b-fp8`       | Qwen3.8-27B (official FP8)   | 48 GB    | **default** for Ada/Hopper 48 GB     |
| `qwen3.8-27b-nvfp4`     | Qwen3.8-27B (NVFP4)          | 32 GB    | Blackwell only, longest context      |
| `qwen3.8-27b-gptq-int4` | Qwen3.8-27B (community int4) | 24 GB    | Ampere and 24-32 GB cards            |
| `qwen3.6-35b-a3b-fp8`   | Qwen3.6-35B-A3B (MoE)        | 48 GB    | fastest generation; verify repo name |
| `olmo3-7b-fp8-16gb`     | Olmo 3 7B Instruct           | 16 GB    | fully open model                     |
| `olmo3-7b-4bit-8gb`     | Olmo 3 7B Instruct           | 8 GB     | RTX 4060-class, short context        |

Full details (RAM, CPU, disk, smallest NVIDIA card, GPU generations): **[docs/HARDWARE.md](docs/HARDWARE.md)**.

## Host requirements

NVIDIA driver, Docker with the Compose plugin (v2.17+ for multiple `--env-file`), and the
NVIDIA Container Toolkit. Linux recommended; Windows works via Docker Desktop + WSL2.

## 1. Prepare (online, once per model)

```bash
# edit config.env first: set VLLM_API_KEY and WEBUI_SECRET_KEY


./scripts/prepare.sh qwen2.5-0.5b-cpu # para pruebas sin GPU
./scripts/prepare.sh qwen3.8-27b-fp8
./scripts/prepare.sh olmo3-7b-fp8-16gb     # any other profiles you want available offline
```

Air-gapped target: run prepare on a connected machine with `--export-images`, copy the
whole project folder plus the `.tar` file, then `docker load -i images-<profile>.tar`.

## 2. Run / switch (offline)

```bash
./scripts/run.sh qwen3.8-27b-fp8        # start
./scripts/run.sh logs                   # ready when "Application startup complete"
./scripts/run.sh olmo3-7b-fp8-16gb   # switch: only vLLM restarts, chats are kept
./scripts/run.sh status
./scripts/run.sh stop
```

Open http://localhost:3000; the first account created becomes admin. The model appears in
the model picker automatically. Other local tools (IDE plugins, scripts) can call
`http://localhost:8000/v1` with `VLLM_API_KEY` and model name `SERVED_MODEL_NAME`.

## Adding a new model

Copy a profile and change `HF_REPO`, `MODEL_DIR`, `SERVED_MODEL_NAME`; set `VLLM_TAG` to the
version the model card recommends and copy its parser flags (`--reasoning-parser`,
`--tool-call-parser`, `--trust-remote-code`). Size it with the formula at the end of
docs/HARDWARE.md. Then `./scripts/prepare.sh <new>` and `./scripts/run.sh <new>`.

## Memory tuning

vLLM reserves `GPU_MEMORY_UTILIZATION` x VRAM at startup: weights first, the rest is KV cache.
On OOM or "not enough KV cache", adjust the profile in this order:

1. lower `MAX_MODEL_LEN` (e.g. 65536 -> 32768)
2. lower `MAX_NUM_SEQS` (concurrent requests)
3. add `--kv-cache-dtype fp8` (Ada/Hopper/Blackwell)
4. add `--enforce-eager` (saves ~0.5-1 GB, slightly slower)
5. switch to a smaller quantization profile

## Offline hardening (already configured)

- vLLM: `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, usage stats off; model loaded from a local path.
- Open WebUI: `OFFLINE_MODE`, web search off, Ollama off, telemetry off, RAG embedding model
  loaded from `models/_embeddings` (downloaded by prepare.sh).

## Troubleshooting

- **vLLM tries to reach huggingface.co**: the model folder is missing `config.json`; re-run prepare.
- **Open WebUI shows no model**: vLLM is still loading (`./scripts/run.sh logs`) or the API keys
  don't match. Admin panel -> Settings -> Connections to re-check.
- **Errors after changing `VLLM_TAG`**: pin it back to the version from the model card.
- **Licenses**: Qwen and Olmo are Apache 2.0.

## Tareas futuras

1. Dar soporte para qwen-code o similar
2. Permitir alternar los modelos y usar uno u otro según la tarea que elija le usuarie.
3. Relajar el modo offline para que pueda acceder a la web
4. Asegurarse de que pueda correr en multiples placas GPU