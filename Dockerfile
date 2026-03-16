FROM runpod/worker-v1-vllm:v2.8.0stable-cuda12.1.0

# Only upgrade transformers for Qwen3.5 architecture support
RUN pip install --no-cache-dir --upgrade transformers huggingface_hub && rm -rf /root/.cache /tmp/*

COPY handler.py /handler.py

ENTRYPOINT []
CMD ["python3", "-u", "/handler.py"]
