FROM runpod/base:0.6.2-cuda12.2.0

# Install vLLM with pre-built wheel (no compilation)
RUN pip install --no-cache-dir \
    vllm \
    runpod \
    && pip cache purge \
    && rm -rf /root/.cache /tmp/*

COPY handler.py /handler.py

ENTRYPOINT []
CMD ["python", "-u", "/handler.py"]
