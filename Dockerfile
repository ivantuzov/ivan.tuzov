FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Download pre-built llama-server with CUDA 12.4 support
RUN curl -L https://github.com/ivantuzov/ivan.tuzov/releases/download/v2.0.4/llama-libs.tar.gz -o /tmp/llama.tar.gz \
    && mkdir -p /opt/llama \
    && tar xzf /tmp/llama.tar.gz -C /opt/llama \
    && cp /opt/llama/llama-server /usr/local/bin/ \
    && cp /opt/llama/*.so* /usr/local/lib/ \
    && ldconfig \
    && rm -rf /tmp/llama.tar.gz /opt/llama

RUN pip install --no-cache-dir runpod huggingface_hub requests

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
