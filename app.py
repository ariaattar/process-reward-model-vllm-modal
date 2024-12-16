import modal
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from openai import OpenAI
from transformers import AutoTokenizer
import subprocess
import time
import asyncio
from huggingface_hub import snapshot_download
import os

app = modal.App("tensorstax-prm-api")
model_volume = modal.Volume.from_name("tensorstax-model-volume-prm", create_if_missing=True)

image = modal.Image.debian_slim().apt_install("git").pip_install(
    "fastapi",
    "vllm==0.6.4.post1",
    "transformers",
    "torch",
    "accelerate",
    "bitsandbytes",
    "huggingface-hub>=0.23.2",
    "hf-transfer",
).run_commands(
    "git clone https://github.com/SkyworkAI/skywork-o1-prm-inference.git",
    "cd skywork-o1-prm-inference && pip install -e ."
)
try:
    from model_utils.io_utils import prepare_input, derive_step_rewards_vllm
except ImportError:
    print("Warning: model_utils.io_utils not found. Skipping import.")

MINUTES = 60
HOURS = 60 * MINUTES

API_KEY_NAME = "X-API-Key" 
API_KEY_1 = "sk_llm_prod_123456789"
API_KEY_2 = "sk_llm_test_987654321"

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header not in [API_KEY_1, API_KEY_2]:
        raise HTTPException(
            status_code=403, detail="Could not validate API key"
        )
    return api_key_header

class GenerationRequest(BaseModel):
    system_prompt: str
    user_prompt: str
    temperature: float = 0.1 

class ScoreRequest(BaseModel):
    problems: list[str]
    responses: list[str]
    model_path: str = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
  
@app.function(
    image=image,
    gpu=modal.gpu.H100(count=1),
    timeout=6 * MINUTES,
    container_idle_timeout=2 * MINUTES,
    allow_concurrent_inputs=200,
    enable_memory_snapshot=True,
    volumes={"/model": model_volume}
)
@modal.asgi_app()
def fastapi_app():

    model_name = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
    model_path = "/model/" + model_name
    weights_path = "/model/weights"
    
    model_volume.reload()
    
    if os.path.exists(weights_path):
        print("Loading model from saved weights...")
        cached_model_path = weights_path
    else:
        print("Downloading model and saving weights...")
        if not os.path.exists(model_path):
            from huggingface_hub import snapshot_download
            snapshot_download(
                model_name,
                local_dir=model_path,
                ignore_patterns=["*.pt", "*.pth", "original/*"],
                force_download=False,
            )
        cached_model_path = model_path
        
        os.makedirs(weights_path, exist_ok=True)
        os.system(f"cp -r {model_path}/* {weights_path}/")
        model_volume.commit()

    process = subprocess.Popen([
        "vllm", "serve", model_path,
        "--host", "0.0.0.0",
        "--port", "8081",
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.9",
        "--enable-prefix-caching",
        "--dtype", "auto",
    ])
    time.sleep(30)
    
    app = FastAPI(title="LLM Generation API")
    
    @app.post("/generate", dependencies=[Depends(get_api_key)])
    async def generate_text(request: GenerationRequest):
        return {"message": "This is a placeholder. Implement your logic here."}

    @app.post("/score", dependencies=[Depends(get_api_key)])
    async def score_responses(request: ScoreRequest):
        try:
            tokenizer = AutoTokenizer.from_pretrained(request.model_path, trust_remote_code=True)
            
            processed_data = [
                prepare_input(problem, response, tokenizer=tokenizer, step_token="\n")
                for problem, response in zip(request.problems, request.responses)
            ]
            input_ids, steps, reward_flags = zip(*processed_data)
            
            client = OpenAI(
                api_key="EMPTY",
                base_url="http://localhost:8081/v1"
            )
            

            rewards = client.embeddings.create(
                input=input_ids,
                model=client.models.list().data[0].id,
            )
            
            step_rewards = derive_step_rewards_vllm(rewards, reward_flags)
            
            return {"step_rewards": step_rewards}
            
        finally:
            pass
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    return app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
