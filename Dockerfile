FROM runpod/worker-v1-vllm:v2.14.0

# Don't upgrade vllm (pulls CUDA 12.9). Only upgrade transformers for Qwen3.5 arch support.
RUN pip install --no-cache-dir --upgrade transformers huggingface_hub && rm -rf /root/.cache /tmp/*

COPY handler.py /handler.py

ENTRYPOINT []
CMD ["python", "-u", "/handler.py"]
