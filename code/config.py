MODELS = {
    'llm': 'cyankiwi/Qwen3.8-27B-AWQ-INT4',
    'llm_vision': 'cyankiwi/Qwen3.8-27B-AWQ-INT4',
    'embedding': 'Qwen/Qwen3-Embedding-0.6B'
}

# Two local vLLM servers, no OpenAI calls (see code/analyze_image.py):
# GPU 0 runs cyankiwi/Qwen3.8-27B-AWQ-INT4 in generate mode (chat + vision).
# GPU 1 runs Qwen3-Embedding-0.6B, a model purpose-built for embeddings (pooling from
# the 27B chat model itself gave noticeably worse FNDDS retrieval quality).
LOCAL_VLLM_BASE_URL = 'http://localhost:8000/v1'
LOCAL_VLLM_EMBED_BASE_URL = 'http://localhost:8001/v1'
