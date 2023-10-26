from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: list

from transformers import AutoTokenizer
import transformers 
import torch

model = "PY007/TinyLlama-1.1B-Chat-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

def converse(prompt,pipeline):
    
    formatted_prompt = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )

    CHAT_EOS_TOKEN_ID = 32002

    sequences = pipeline(
        formatted_prompt,
        do_sample=True,
        top_k=50,
        top_p = 0.9,
        num_return_sequences=1,
        repetition_penalty=1.1,
        max_new_tokens=1024,
        eos_token_id=CHAT_EOS_TOKEN_ID,
    )
    
    return [f"Result: {seq['generated_text']}" for seq in sequences]

@app.post("/chat/")
async def chat_with_bot(chat_request: ChatRequest):
    response = converse(chat_request.prompt, pipeline)
    return {"response": response}
