# Requerimientos de hardware por perfil

Todos los números en este archivo son **estimativos** y se derivan de tener en cuenta el tamaño de los modelos (su archivo de pesos), el uso de [un caché clave valor (KV)](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) y los paso-a-paso oficiales publicados para cada modelo.

No son un benchmark, sino una aproximación de lo que podría requerir para el uso normal de 1-2 usuaries concurrentes.

## Tabla de perfiles

| Perfil                  | Pesos en GPU | VRAM mínima             | RAM mínima | vCPU mín. | Disco  | Contexto a VRAM mínima            |
|-------------------------|--------------|-------------------------|------------|-----------|--------|-----------------------------------|
| `qwen3.8-27b-nvfp4`     | ~25 GB       | 32 GB (solo Blackwell)  | 32 GB      | 4         | ~30 GB | 32K en 32 GB; hasta 128K en 48 GB |
| `qwen3.8-27b-fp8`       | ~27 GB       | 48 GB (32 GB = ~8K ctx) | 32 GB      | 4         | ~30 GB | 64K en 48 GB                      |
| `qwen3.8-27b-gptq-int4` | ~18 GB       | 24 GB                   | 32 GB      | 4         | ~20 GB | ~16-32K en 24 GB; 64K+ en 48 GB   |
| `qwen3.6-35b-a3b-fp8`   | ~35 GB       | 48 GB                   | 48 GB      | 4         | ~37 GB | ~32K                              |
| `qwen2.5-0.5b-cpu`      | -            | (solo CPU)              | 4 GB       | 2         | ~2 GB  | 16K                               |
| `olmo3-7b-fp8-16gb`     | ~7.5 GB      | 16 GB (Ada o posterior) | 16 GB      | 4         | ~15 GB | ~32K                              |
| `olmo3-7b-4bit-8gb`     | ~5 GB        | 8 GB                    | 16 GB      | 4         | ~15 GB | ~4-8K                             |
| `smollm2-135m-cpu`      | -            | (solo CPU)              | 2 GB       | 2         | ~1 GB  | 8K                                |

Notas sobre las columnas:

- **Pesos en GPU**: lo que carga vLLM; el resto de la VRAM se convierte en caché KV
  (memoria de conversación). Más VRAM libre = contexto más largo y más usuaries simultánees.
- **RAM mínima**: vLLM lee los pesos completos por RAM antes de cargarlos en la GPU; los
  perfiles que cuantifican al cargar (Olmo) leen el checkpoint BF16 completo (~15 GB)
  primero. Sumá ~2-4 GB para Open WebUI y el sistema operativo.
- **vCPU mín.**: la GPU hace el trabajo pesado. vLLM necesita ~1-2 núcleos para
  scheduling, tokenización y la API; Open WebUI necesita 1-2 más (los embeddings de RAG
  corren en CPU). 4 es el mínimo, 8 es cómodo. La cantidad de CPUs casi no afecta la
  velocidad de generación.
- **Disco**: solo los archivos del modelo. Sumá ~20-25 GB por versión de imagen de vLLM y
  ~5 GB para Open WebUI. Usá un SSD/NVMe; el tiempo de carga lo domina la velocidad de
  lectura del disco.

## Placa NVIDIA mínima por perfil

| Perfil                  | Placa mínima                                                                  | Placa recomendada                      |
|-------------------------|-------------------------------------------------------------------------------|----------------------------------------|
| `qwen3.8-27b-nvfp4`     | RTX PRO 4500 Blackwell (32 GB) o RTX 5090 (32 GB)                             | RTX PRO 5000 Blackwell (48 GB)         |
| `qwen3.8-27b-fp8`       | RTX 6000 Ada (48 GB) / L40S (48 GB)                                           | ídem; RTX PRO 5000 Blackwell (48 GB)   |
| `qwen3.8-27b-gptq-int4` | RTX 4500 Ada (24 GB), RTX PRO 4000 Blackwell (24 GB), RTX 4090 / 3090 (24 GB) | RTX 5000 Ada (32 GB) o cualquier 48 GB |
| `qwen3.6-35b-a3b-fp8`   | RTX 6000 Ada (48 GB) / L40S (48 GB)                                           | RTX PRO 5000 Blackwell (48 GB)         |
| `olmo3-7b-fp8-16gb`     | RTX 4060 Ti 16 GB, RTX 4000 Ada (20 GB)                                       | RTX 4500 Ada (24 GB)                   |
| `olmo3-7b-4bit-8gb`     | RTX 4060 (8 GB)                                                               | preferir el perfil FP8 en 16 GB+       |

## Generaciones compatibles

| Generación                                        | FP8                | NVFP4  | Mejor perfil Qwen3.8                        |
|---------------------------------------------------|--------------------|--------|---------------------------------------------|
| Ampere (RTX 3090, A100)                           | emulado, más lento | no     | `qwen3.8-27b-gptq-int4`                     |
| Ada (RTX 6000 Ada, L40S, RTX 4090/4060)           | nativo             | no     | `qwen3.8-27b-fp8` (48 GB) o GPTQ (24-32 GB) |
| Hopper (H100, H200)                               | nativo             | no     | `qwen3.8-27b-fp8`                           |
| Blackwell (RTX PRO 4000/4500/5000/6000, RTX 5090) | nativo             | nativo | `qwen3.8-27b-nvfp4`                         |

Driver mínimo de NVIDIA: el que requiera la versión de CUDA de la imagen de vLLM elegida
(las imágenes recientes necesitan driver 570+ para CUDA 12.8+). Revisá las notas de la
imagen al cambiar `VLLM_TAG`.

## Cómo estimar un modelo nuevo

```
pesos_GB ~= parámetros_en_miles_de_millones x bytes_por_parámetro
            (BF16 = 2, FP8 = 1, 4-bit = ~0.55 incluyendo overhead)
VRAM necesaria ~= pesos_GB + 1.5 GB (CUDA/activaciones) + caché KV
caché KV por token (bytes) ~= 2 x capas x kv_heads x head_dim x bytes_por_valor_kv
```

Para modelos MoE, usá el total de parámetros para la memoria (todos los expertos se
cargan), no los parámetros activos. Los modelos con atención híbrida/lineal (Qwen3.8)
necesitan mucho menos caché KV que lo que sugiere la fórmula, porque solo algunas capas
mantienen un caché completo.
