FROM vllm/vllm-openai:latest

# vLLM image already has vllm, torch, CUDA — just add runpod
RUN pip install --no-cache-dir runpod

# Copy handler
COPY handler.py /handler.py

# RunPod serverless entrypoint
CMD ["python", "-u", "/handler.py"]
