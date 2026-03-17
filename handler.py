"""
RunPod Serverless Handler for llama.cpp with Qwen3.5.
"""
import os
import subprocess
import time
import requests
import runpod
from huggingface_hub import hf_hub_download

MODEL_REPO = os.environ.get("MODEL_REPO", "mradermacher/Huihui-Qwen3.5-27B-abliterated-GGUF")
MODEL_FILE = os.environ.get("MODEL_FILE", "Huihui-Qwen3.5-27B-abliterated.Q4_K_M.gguf")
MODEL_DIR = os.environ.get("MODEL_DIR", "/runpod-volume/models")
CTX_SIZE = int(os.environ.get("CTX_SIZE", "8192"))
GPU_LAYERS = int(os.environ.get("GPU_LAYERS", "99"))
PORT = 8080

server_process = None
server_ready = False


def ensure_model():
    """Download model if not present."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    if not os.path.exists(model_path):
        print(f"[Handler] Downloading {MODEL_REPO}/{MODEL_FILE}...")
        hf_hub_download(MODEL_REPO, MODEL_FILE, local_dir=MODEL_DIR)
        print(f"[Handler] Download complete: {model_path}")
    return model_path


def start_server():
    """Start llama-server."""
    global server_process, server_ready
    model_path = ensure_model()

    print(f"[Handler] Starting llama-server: {MODEL_FILE}, ctx={CTX_SIZE}, gpu_layers={GPU_LAYERS}")
    server_process = subprocess.Popen([
        "llama-server",
        "--model", model_path,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--n-gpu-layers", str(GPU_LAYERS),
        "--ctx-size", str(CTX_SIZE),
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Wait for server to be ready
    for i in range(120):
        try:
            r = requests.get(f"http://localhost:{PORT}/health", timeout=2)
            if r.status_code == 200:
                server_ready = True
                print(f"[Handler] Server ready after {i+1}s")
                return
        except:
            pass
        time.sleep(1)
    print("[Handler] Server failed to start in 120s")


def handler(job):
    """Process inference request."""
    global server_ready
    if not server_ready:
        start_server()
        if not server_ready:
            return {"error": "Server failed to start"}

    job_input = job["input"]

    # Health check
    if job_input.get("ping"):
        return {"status": "ok", "model": MODEL_FILE}

    # Forward to llama-server OpenAI API
    messages = job_input.get("messages", [])
    if not messages and job_input.get("openai_input"):
        params = job_input["openai_input"]
        messages = params.get("messages", [])
    elif not messages and job_input.get("prompt"):
        messages = [{"role": "user", "content": job_input["prompt"]}]

    if not messages:
        return {"error": "No messages provided"}

    body = {
        "model": MODEL_FILE,
        "messages": messages,
        "max_tokens": job_input.get("max_tokens") or job_input.get("openai_input", {}).get("max_tokens", 2000),
        "temperature": job_input.get("temperature") or job_input.get("openai_input", {}).get("temperature", 0.7),
    }

    try:
        r = requests.post(f"http://localhost:{PORT}/v1/chat/completions", json=body, timeout=300)
        data = r.json()

        # Pass through raw response - let client handle parsing
        msg = data.get("choices", [{}])[0].get("message", {})
        content = msg.get("content", "")
        reasoning = msg.get("reasoning_content", "")

        return {
            "choices": [{
                "message": {"role": "assistant", "content": content},
                "finish_reason": data.get("choices", [{}])[0].get("finish_reason", "stop"),
            }],
            "usage": data.get("usage", {}),
            "model": MODEL_FILE,
        }
    except Exception as e:
        return {"error": str(e)}


# Start server on cold start
print("[Handler] Initializing...")
start_server()

runpod.serverless.start({"handler": handler})
