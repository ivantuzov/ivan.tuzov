FROM runpod/base:0.6.2-cuda12.2.0

ENV DEBIAN_FRONTEND=noninteractive

# Install vLLM with CUDA support
RUN pip install --no-cache-dir \
    vllm>=0.9.0 \
    runpod \
    huggingface_hub \
    && pip cache purge

# Copy handler
COPY handler.py /handler.py

# RunPod serverless entrypoint
CMD ["python", "-u", "/handler.py"]
