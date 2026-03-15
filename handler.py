"""
RunPod Serverless Handler for vLLM with OpenAI-compatible API.
Supports Qwen3.5 and other models via transformers fallback.
"""

import os
import runpod
from vllm import LLM, SamplingParams

# Load model on cold start
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-27B-GPTQ-Int4")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.92"))
QUANTIZATION = os.environ.get("QUANTIZATION", None)
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "1") == "1"
DTYPE = os.environ.get("DTYPE", "auto")

print(f"[Handler] Loading model: {MODEL_NAME}")
print(f"[Handler] Max model len: {MAX_MODEL_LEN}, GPU util: {GPU_MEMORY_UTILIZATION}")

llm_kwargs = dict(
    model=MODEL_NAME,
    max_model_len=MAX_MODEL_LEN,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    trust_remote_code=TRUST_REMOTE_CODE,
    dtype=DTYPE,
    enforce_eager=False,
)

if QUANTIZATION and QUANTIZATION.lower() not in ("none", ""):
    llm_kwargs["quantization"] = QUANTIZATION.lower()

# Try native vLLM first, fall back to transformers backend for unsupported architectures
try:
    llm = LLM(**llm_kwargs)
    print("[Handler] Model loaded with native vLLM backend")
except ValueError as e:
    if "not supported" in str(e):
        print(f"[Handler] Native vLLM doesn't support this model, using transformers backend...")
        llm_kwargs["model_impl"] = "transformers"
        llm = LLM(**llm_kwargs)
        print("[Handler] Model loaded with transformers backend")
    else:
        raise

tokenizer = llm.get_tokenizer()
print(f"[Handler] Model loaded successfully!")


def handler(job):
    """Process a single inference request."""
    job_input = job["input"]

    # Support OpenAI-compatible format
    openai_input = job_input.get("openai_input", None)

    if openai_input or "messages" in job_input:
        # Chat completions mode
        params = openai_input or job_input
        messages = params.get("messages", [])
        max_tokens = params.get("max_tokens", 512)
        temperature = params.get("temperature", 0.7)
        top_p = params.get("top_p", 0.9)
        top_k = params.get("top_k", -1)
        stop = params.get("stop", None)
        presence_penalty = params.get("presence_penalty", 0.0)
        frequency_penalty = params.get("frequency_penalty", 0.0)

        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )

        outputs = llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        usage = {
            "prompt_tokens": len(outputs[0].prompt_token_ids),
            "completion_tokens": len(outputs[0].outputs[0].token_ids),
            "total_tokens": len(outputs[0].prompt_token_ids) + len(outputs[0].outputs[0].token_ids),
        }

        return {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text,
                    },
                    "finish_reason": outputs[0].outputs[0].finish_reason,
                }
            ],
            "usage": usage,
            "model": MODEL_NAME,
        }

    elif "prompt" in job_input:
        prompt = job_input["prompt"]
        max_tokens = job_input.get("max_tokens", 512)
        temperature = job_input.get("temperature", 0.7)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
        )

        outputs = llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text

        return {"text": generated_text, "model": MODEL_NAME}

    else:
        return {"error": "Provide 'messages' (chat) or 'prompt' (completion)"}


runpod.serverless.start({"handler": handler})
