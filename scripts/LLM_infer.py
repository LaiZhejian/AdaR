import json
import yaml
import torch
import os
import sys
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm
import openai
import re

cfg = None
client = None

def remove_think_content(text: str) -> str:
    # 使用非贪婪匹配，删除 <think>...</think> 之间的内容
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned_text.strip()

async def process_prompt_async(data_item, sampling_params):
    while True:
        try:
            completion = await client.chat.completions.create(
                model="default",
                messages=data_item["messages"] if isinstance(data_item["messages"], list) else [{"role": "user", "content": data_item["messages"]}],
                timeout=3600,
                **sampling_params,
            )
            data_item['generated_texts'] = [remove_think_content(gen.message.content) for gen in completion.choices]
            return data_item
        except Exception as e:
            # pass
            print(f"Retry: {e}", file=sys.stderr)
            await asyncio.sleep(1)
    data_item["generated_texts"] = ["ERROR"]
    return data_item


async def process_in_parallel(data, sampling_params, output_path, batch_size=50):

    global client
    client = openai.AsyncClient(base_url=cfg["process"][sys.argv[1]]["url"], api_key=cfg["process"][sys.argv[1]]["api_key"])
    sem = asyncio.Semaphore(batch_size)  # Control the maximum concurrency
    results_buffer = []
    total = len(data)

    async def sem_task(item):
        async with sem:
            return await process_prompt_async(item, sampling_params)

    with tqdm(total=total, desc="Inferring") as pbar, open(output_path, "a", encoding="utf-8") as out_file:
        tasks = [asyncio.create_task(sem_task(item)) for item in data]

        batch = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            batch.append(result)
            pbar.update(1)

            if len(batch) >= batch_size:
                out_file.write("\n".join(json.dumps(r, ensure_ascii=False) for r in batch) + "\n")
                batch.clear()

        if batch:
            out_file.write("\n".join(json.dumps(r, ensure_ascii=False) for r in batch) + "\n")
            out_file.flush()

    print()
    print("="* 40)
    print(f"Results for {sys.argv[1]} have been generated!\n")
    print(f"Total: {len(data)} instances")
    print(f"Output path: {output_path}")
    print(f'Deployed url: {cfg["process"][sys.argv[1]]["url"]}')
    print(f'Sampling Params: {json.dumps(sampling_params, ensure_ascii=False)}')
    print(f'Inference batch size: {cfg["process"][sys.argv[1]]["bsz"]}')
    print("="* 40)
    print()



async def process_local(data, sampling_params, output_path):
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import BeamSearchParams
    
    if cfg["process"][sys.argv[1]]["generate_model_path"] != -1:
        model_path = cfg["process"][sys.argv[1]]["generate_model_path"]
    
    # 配置tensor并行
    tensor_parallel_size = cfg["process"][sys.argv[1]]["tensor_parallel_size"]
    if tensor_parallel_size == -1:
        tensor_parallel_size = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(','))
    
    # 初始化LLM
    llm = LLM(
        model=model_path,
        dtype=torch.bfloat16,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=cfg["process"][sys.argv[1]]["gpu_memory_utilization"]
    )
    
    from transformers import AutoTokenizer
    tokenizer= AutoTokenizer.from_pretrained(model_path, use_fast=False)

    prompts = [tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=True) for item in data]
    outputs = llm.generate(prompts, SamplingParams(**sampling_params))
    
    with open(output_path, "w", encoding="utf-8") as f:
        for i, output in enumerate(outputs):
            data[i]['generated_texts'] = [remove_think_content(gen.text) for gen in output.outputs]
            f.write(json.dumps(data[i], ensure_ascii=False) + "\n")

    print()
    print("="* 40)
    print("Results for template and code generation have been generated!")
    print(f"Total: {len(data)} instances")
    print(f"Output path: {output_path}")
    print(f'Deployed model: {os.path.basename(model_path)}')
    print("="* 40)
    print()


async def main():
    global cfg
    with open('config.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
        
    from utils import set_seed
    set_seed(cfg["process"]["seed"])

    if sys.argv[1] == "template_and_code_generation":
        input_path = os.path.join(cfg["process"]["tmp_folder"], sys.argv[1], f'{cfg["data"]["dataset_name"]}_prompt.jsonl')
        output_path = os.path.join(cfg["process"]["tmp_folder"], sys.argv[1], f'{cfg["data"]["dataset_name"]}_generated.jsonl')
    else:
        input_path = os.path.join(cfg["process"]["tmp_folder"], sys.argv[1], f'{cfg["data"]["dataset_name"]}_{"|".join(str(item) for item in cfg["process"]["controllable_perturbation"]["alpha_list"])}_{cfg["process"]["controllable_perturbation"]["sample_times"]}_prompt.jsonl')
        output_path = os.path.join(cfg["process"]["tmp_folder"], sys.argv[1], f'{cfg["data"]["dataset_name"]}_{"|".join(str(item) for item in cfg["process"]["controllable_perturbation"]["alpha_list"])}_{cfg["process"]["controllable_perturbation"]["sample_times"]}_generated.jsonl')
       
    is_parallel = cfg["process"][sys.argv[1]]["is_parallel"]
    
    
    data  = []

    with open(input_path, "r", encoding="utf-8") as f:   
        for line in f:
            data_item = json.loads(line)
            data.append(data_item)
    
    sampling_params = cfg["process"][sys.argv[1]]['sampling-params']

    if is_parallel:
        await process_in_parallel(
            data=data,
            sampling_params=sampling_params,
            output_path=output_path,
            batch_size=cfg["process"][sys.argv[1]]["bsz"]
        )
                    
    else:
        await process_local(
            data, 
            sampling_params, 
            output_path
        )

if __name__ == "__main__":
    asyncio.run(main())
