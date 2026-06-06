!/bin/bash

python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --enable-lora \
  --lora-modules news-analyzer=Shahdddddd/news-analyzer \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype float16 \
  --quantization bitsandbytes \
  --gpu-memory-utilization 0.8 \
  --max-model-len 4000 \
  --max-lora-rank 64 \
  --enforce-eager \
  --max-cpu-loras 1