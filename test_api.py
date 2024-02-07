import datetime
import json

import torch
import uvicorn
from fastapi import FastAPI, Request
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    LlamaTokenizer,
    LlamaForCausalLM,
)

MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}
import logging
from loguru import logger
logger.add('logs/query_api.log')


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

current_time = datetime.datetime.now()

app = FastAPI()

prompt_input = (
    "{instruction}\n### Response:\n"
)

def generate_prompt(instruction, input=None):
    if input:
        instruction = instruction + '\n' + input
    return prompt_input.format_map({'instruction': instruction})


@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    raw_input_text = json_post_list.get('prompt')

    # input_text = tokenizer.bos_token + generate_prompt(instruction=raw_input_text) + tokenizer.eos_token
    input_text = generate_prompt(instruction=raw_input_text)
    
    inputs = tokenizer(input_text, return_tensors="pt")
    generate_kwargs = dict(
        input_ids=inputs["input_ids"].to('cuda:0'),
        max_new_tokens=4096,
        temperature=0.2,
        do_sample=True,
        top_p=1.0,
        top_k=10,
        repetition_penalty=1.1,
    )
    generation_output = model.generate(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **generate_kwargs
    )

    s = generation_output[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    response = output.split("### Response:")[1].strip()

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response.replace('ChatGLM-6B', '').replace('chatglm-6b', ''),
        "status": 200,
        "time": time
    }
    log = "===>[" + time + "]\n " + '", prompt:"\n' + raw_input_text + '", response:"\n' + repr(response) + '"'
    logger.info(log)
    torch_gc()
    return answer

if __name__ == '__main__':
    model_name = 's2_Llama-2-7b-chat-hf'
    model_name = 'Llama-2-7b-chat-hf'
    model_type = 'llama'



    model_class, tokenizer_class = MODEL_CLASSES[model_type]
    tokenizer = tokenizer_class.from_pretrained(model_name, trust_remote_code=True)
    model = model_class.from_pretrained(model_name, trust_remote_code=True, device_map='cuda:0').half()
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=2002, workers=1)
