FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl cmake build-essential git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Build llama.cpp with CUDA
RUN git clone https://github.com/ggml-org/llama.cpp /opt/llama.cpp && \
    cd /opt/llama.cpp && \
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" && \
    cmake --build build --config Release -j$(nproc) && \
    cp build/bin/llama-server /usr/local/bin/ && \
    rm -rf /opt/llama.cpp

RUN pip install --no-cache-dir runpod huggingface_hub requests

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
