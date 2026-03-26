# Running GigaChat3.1-10B-A1.8B on Mac with 36 GB unified memory

## Backend: llama.cpp (recommended for Apple Silicon)

Transformers + `device_map="auto"` does NOT use Metal GPU on Mac — it falls back to CPU.
llama.cpp has native Metal support and will be significantly faster.

### 1. Build llama.cpp with Metal

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release --target llama-server -j $(sysctl -n hw.logicalcpu)
```

### 2. Download the GGUF variant

With 36 GB unified memory you can comfortably run the BF16 variant (21.4 GB):

| Variant  | Size    | Recommendation                  |
|----------|---------|---------------------------------|
| BF16     | 21.4 GB | **Best quality** — use this     |
| Q8_0     | 11.4 GB | Minimal quality loss, faster    |
| Q6_K     | 8.78 GB | Good balance                    |
| Q4_K_M   | 6.47 GB | Skip — you have the RAM         |

```bash
huggingface-cli download ai-sage/GigaChat3.1-10B-A1.8B-GGUF \
  --include "gigachat3.1-10b-a1.8b-bf16.gguf" \
  --local-dir ./cache
```

### 3. Run the server

```bash
./build/bin/llama-server \
    -m ./cache/gigachat3.1-10b-a1.8b-bf16.gguf \
    -np 1 \
    -cb \
    -ctk q8_0 \
    -ctv q8_0 \
    -fa on \
    --n-gpu-layers 999 \
    --ctx-size 32768 \
    --port 8080 \
    --host 0.0.0.0 \
    --jinja
```

`--n-gpu-layers 999` offloads all layers to Metal GPU.
With 36 GB unified memory the full BF16 model fits entirely on-chip.

### 4. Send a request

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 1000,
    "temperature": 0
  }'
```

---

## Alternative: transformers with MPS

Add MPS device explicitly — `device_map="auto"` alone won't pick Metal:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="mps",
    cache_dir=cache_dir,
)
```

Slower than llama.cpp but requires no separate build step.
