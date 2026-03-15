FROM vllm/vllm-openai:latest

# vLLM image already has vllm, torch, CUDA — just add runpod
RUN pip install --no-cache-dir runpod

# Copy handler
COPY handler.py /handler.py

# Override vllm entrypoint
ENTRYPOINT []
CMD ["python", "-u", "/handler.py"]
