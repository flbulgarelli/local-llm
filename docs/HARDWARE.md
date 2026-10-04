# Requerimientos de hardware por perfil

Todos los números en este archivo son **estimativos** y se derivan de tener en cuenta el tamaño de los modelos (su archivo de pesos), el uso de [un caché clave valor (KV)](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) y los paso-a-paso oficiales publicados para cada modelo.

No son un benchmark, sino una aproximación de lo que podría requerir para el uso normal de 1-2 usuaries concurrentes.

## Reference machine

| Resource   | Value                 |
|------------|-----------------------|
| GPU        | 1x NVIDIA, 48 GB VRAM |
| System RAM | 64 GiB                |
| vCPUs      | 8                     |

**Every profile in this project runs on the reference machine.** Pick the Qwen FP8,
NVFP4 or GPTQ variant according to the GPU generation (see "GPU generation" below).

## Summary table

| Profile                            | Weights on GPU | Min VRAM                     | Min system RAM | Min vCPU | Disk   | Context at min VRAM | Fits reference machine            |
|------------------------------------|----------------|------------------------------|----------------|----------|--------|---------------------|-----------------------------------|
| `qwen3.8-27b-nvfp4`                | ~25 GB         | 32 GB (Blackwell only)       | 32 GB          | 4        | ~30 GB | 32K+                | Yes (if Blackwell), up to 128K    |
| `qwen3.8-27b-fp8`                  | ~27 GB         | 48 GB (32 GB = ~8K ctx only) | 32 GB          | 4        | ~30 GB | 64K on 48 GB        | Yes, 64K                          |
| `qwen3.8-27b-gptq-int4`            | ~18 GB         | 24 GB                        | 32 GB          | 4        | ~20 GB | ~16-32K             | Yes, 64K+                         |
| `qwen3.6-35b-a3b-fp8`              | ~35 GB         | 48 GB                        | 48 GB          | 4        | ~37 GB | ~32K                | Yes, 32K                          |
| `olmo3-7b-fp8-16gb`                | ~7.5 GB        | 16 GB (Ada or newer)         | 16 GB          | 4        | ~15 GB | ~32K                | Yes                               |
| `olmo3-7b-4bit-8gb`                | ~5 GB          | 8 GB                         | 16 GB          | 4        | ~15 GB | ~4-8K               | Yes (use the FP8 profile instead) |
| `qwen2.5-0.5b-cpu`                 | —              | **no GPU** (CPU only)        | 4 GB           | 2        | ~2 GB  | 4K                  | Yes (infrastructure testing only) |

Notes on the columns:

- **Weights on GPU**: what vLLM loads; the rest of the VRAM becomes KV cache
  (conversation memory). More free VRAM = longer context and more simultaneous users.
- **Min system RAM**: vLLM streams weights from disk, but reads whole shards through
  RAM; profiles that quantize at load time (Olmo) read the full BF16 checkpoint
  (~15 GB) first. Add ~2-4 GB for Open WebUI and the OS. 64 GiB covers every profile.
- **Min vCPU**: the GPU does the heavy work. vLLM needs ~1-2 cores for scheduling,
  tokenization and the API; Open WebUI needs 1-2 more (RAG embeddings run on CPU).
  4 is the floor, 8 is comfortable. CPU count barely affects generation speed.
- **Disk**: model files only. Add ~20-25 GB per vLLM image version and ~5 GB for
  Open WebUI. Use an SSD/NVMe; load time is dominated by disk read speed.

## Smallest NVIDIA card per profile

| Profile                            | Minimum card                                                                  | Recommended card                       |
|------------------------------------|-------------------------------------------------------------------------------|----------------------------------------|
| `qwen3.8-27b-nvfp4`                | RTX PRO 4500 Blackwell (32 GB) or RTX 5090 (32 GB)                            | RTX PRO 5000 Blackwell (48 GB)         |
| `qwen3.8-27b-fp8`                  | RTX 6000 Ada (48 GB) / L40S (48 GB)                                           | same; RTX PRO 5000 Blackwell (48 GB)   |
| `qwen3.8-27b-gptq-int4`            | RTX 4500 Ada (24 GB), RTX PRO 4000 Blackwell (24 GB), RTX 4090 / 3090 (24 GB) | RTX 5000 Ada (32 GB) or any 48 GB card |
| `qwen3.6-35b-a3b-fp8`              | RTX 6000 Ada (48 GB) / L40S (48 GB)                                           | RTX PRO 5000 Blackwell (48 GB)         |
| `olmo3-7b-fp8-16gb`                | RTX 4060 Ti 16 GB, RTX 4000 Ada (20 GB)                                       | RTX 4500 Ada (24 GB)                   |
| `olmo3-7b-4bit-8gb`                | RTX 4060 (8 GB)                                                               | use the FP8 profile on 16 GB+          |


## GPU generation matters as much as VRAM

| Generation (examples)                             | FP8              | NVFP4  | Best Qwen3.8 profile                         |
|---------------------------------------------------|------------------|--------|----------------------------------------------|
| Ampere (RTX A6000, RTX 3090, A100)                | emulated, slower | no     | `qwen3.8-27b-gptq-int4`                      |
| Ada (RTX 6000 Ada, L40S, RTX 4090/4060)           | native           | no     | `qwen3.8-27b-fp8` (48 GB) or GPTQ (24-32 GB) |
| Hopper (H100, H200)                               | native           | no     | `qwen3.8-27b-fp8`                            |
| Blackwell (RTX PRO 4000/4500/5000/6000, RTX 5090) | native           | native | `qwen3.8-27b-nvfp4`                          |

Minimum NVIDIA driver: whatever the chosen vLLM image's CUDA version requires
(recent vLLM images need a 570+ series driver for CUDA 12.8+). Check the image notes
when changing `VLLM_TAG`.

## Olmo 3 7B on an RTX 4060 (8 GB)

It works but is tight: ~5 GB of 4-bit weights, ~1 GB CUDA overhead, and the
remaining ~1.5 GB of KV cache holds roughly 4-8K tokens (Olmo 3 7B uses 32 full KV
heads, so its KV cache per token is large for its size; FP8 KV cache halves it).
Expect one or two concurrent chats. If the card also drives a monitor, lower
`GPU_MEMORY_UTILIZATION` further (0.80) or `MAX_MODEL_LEN` to 4096.

On Windows, this stack runs through Docker Desktop with the WSL2 backend, which
supports NVIDIA GPUs; allocate at least 16 GB RAM to WSL (`.wslconfig`).

Olmo 3 7B is a general-purpose model and noticeably weaker at code than Qwen3.8-27B.
Its main advantages are size and being fully open (weights, data and training code).

## How to estimate a new model

```
weights_GB ~= parameters_in_billions x bytes_per_param
              (BF16 = 2, FP8 = 1, 4-bit = ~0.55 incl. overhead)
VRAM needed ~= weights_GB + 1.5 GB (CUDA/activations) + KV cache
KV cache per token (bytes) ~= 2 x layers x kv_heads x head_dim x bytes_per_kv_value
```

For MoE models, use **total** parameters for memory (all experts must be loaded),
not active parameters. Hybrid/linear-attention models (Qwen3.8) need much
less KV cache than the formula suggests, because only some layers keep a full cache.
