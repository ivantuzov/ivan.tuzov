FROM runpod/worker-v1-vllm:v2.14.0

COPY handler.py /handler.py

ENTRYPOINT []
CMD ["python3", "-u", "/handler.py"]
