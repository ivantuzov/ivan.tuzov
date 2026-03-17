FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git cmake build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Clone llama.cpp
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp

# Build with CUDA
WORKDIR /opt/llama.cpp
RUN cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"
RUN cmake --build build --config Release -j$(nproc)
RUN cp build/bin/llama-server /usr/local/bin/

# Cleanup build files
WORKDIR /
RUN rm -rf /opt/llama.cpp

# Install Python dependencies
RUN pip install --no-cache-dir runpod huggingface_hub requests

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
