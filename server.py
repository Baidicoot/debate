from typing import List, Literal, Optional, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
import time
import uvicorn
import asyncio
import logging
import traceback
from fastapi.responses import JSONResponse
import os

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="vLLM OpenAI-compatible API with LoRA support")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
BASE_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LORAS_PATH = "adapters/llama-3.1-8b-instruct"
MAX_LORAS = 16
engine = None
tokenizer = None
loras = {}

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str # in this case, used to specify the LoRA adapter
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{time.time()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[dict]
    usage: dict

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handler caught: {exc}")
    logger.error(f"Request path: {request.url.path}")
    logger.error(f"Traceback: {''.join(traceback.format_tb(exc.__traceback__))}")
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
            "traceback": traceback.format_exc()
        }
    )

@app.on_event("startup")
def startup_event():
    global engine, tokenizer
    try:
        logger.info("Initializing tokenizer and vLLM engine...")
        
        # Initialize tokenizer from specific model
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

        num_loras = min(MAX_LORAS, len(os.listdir(LORAS_PATH)))
        # Initialize engine with specific tokenizer
        engine_args = AsyncEngineArgs(
            model=BASE_MODEL_NAME,
            tokenizer=tokenizer,
            tensor_parallel_size=1,
            max_num_batched_tokens=8192,
            max_model_len=6144,
            trust_remote_code=True,
            max_loras=num_loras,
            max_lora_rank=64,
        )
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("Initialization completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        logger.error(traceback.format_exc())
        raise

def get_lora_index(lora_name: str):
    num_loras = len(loras)
    if lora_name not in loras:
        if num_loras >= MAX_LORAS:
            raise ValueError(f"Maximum number of LoRAs ({MAX_LORAS}) reached")
        loras[lora_name] = num_loras
    return loras[lora_name]

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        logger.info(f"Received chat completion request for model: {request.model}")

        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens if request.max_tokens else 2048,
            stop=request.stop if request.stop else None,
        )
        lora_index = get_lora_index(request.model)
        lora_request = LoRARequest(
            request.model,
            lora_index,
            os.path.join(LORAS_PATH, request.model)
        )

        # Convert messages to the model's chat format using the tokenizer
        # Get the chat template and apply it
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate completion using AsyncLLMEngine
        request_id = f"request_{int(time.time() * 1000)}"
        final_result = None
        async for result in engine.generate(
            prompt,
            sampling_params,
            request_id=request_id,
            lora_request=lora_request
        ):
            final_result = result  # Keep overwriting until we get the final result

        # Format response
        choices = []
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0

        if final_result:
            generated_text = final_result.outputs[0].text.replace("<|eot_id|>", "")
            prompt_tokens = len(final_result.prompt_token_ids)
            completion_tokens = len(final_result.outputs[0].token_ids)
            total_tokens = prompt_tokens + completion_tokens

            choice = {
                "index": len(choices),
                "message": {
                    "role": "assistant",
                    "content": generated_text.strip()
                },
                "finish_reason": "stop" if result.outputs[0].finish_reason == "stop" else "length"
            }
            choices.append(choice)

        response = {
            "model": request.model,
            "choices": choices,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }
        return ChatCompletionResponse(**response)

    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        logger.error(f"Detailed traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
        )

if __name__ == "__main__":
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        log_level="debug",
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "()": "uvicorn.logging.DefaultFormatter",
                    "fmt": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "use_colors": True,
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
            },
            "loggers": {
                "": {"handlers": ["default"], "level": "DEBUG"},
            },
        },
    )