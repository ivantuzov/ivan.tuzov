FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential ca-certificates

RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp /build/llama.cpp

WORKDIR /build/llama.cpp
RUN cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86" -DLLAMA_CURL=OFF -DGGML_NATIVE=OFF
RUN cmake --build build --target llama-server -j2

# Runtime stage - much smaller
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the server binary and required libs
COPY --from=builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/
COPY --from=builder /build/llama.cpp/build/src/*.so* /usr/local/lib/
COPY --from=builder /build/llama.cpp/build/ggml/src/*.so* /usr/local/lib/
RUN ldconfig

RUN pip install --no-cache-dir runpod huggingface_hub requests

COPY handler.py /handler.py

CMD ["python3", "-u", "/handler.py"]
