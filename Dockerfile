FROM runpod/worker-v1-vllm:v2.14.0

# This image already has vllm + runpod + CUDA
# Just upgrade vllm to latest for Qwen3.5 support
RUN pip install --no-cache-dir --upgrade vllm && rm -rf /root/.cache /tmp/*

COPY handler.py /handler.py

ENTRYPOINT []
CMD ["python", "-u", "/handler.py"]
