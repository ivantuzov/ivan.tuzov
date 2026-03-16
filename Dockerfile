FROM runpod/worker-v1-vllm:v2.8.0stable-cuda12.1.0

COPY handler.py /handler.py

ENTRYPOINT []
CMD ["python3", "-u", "/handler.py"]
