from transformers import AutoTokenizer
import json
import re
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import yaml
import os
import random

with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

from utils import set_seed
set_seed(cfg["process"]["seed"])

input_path = os.path.join(cfg["process"]["tmp_folder"], "query_paraphrase", f'{cfg["data"]["dataset_name"]}_{"|".join(str(item) for item in cfg["process"]["controllable_perturbation"]["alpha_list"])}_{cfg["process"]["controllable_perturbation"]["sample_times"]}_generated.jsonl')
paths = {k: os.path.join(cfg["target_dir"], f'{cfg["data"]["dataset_name"]}_aug_{k}_{cfg["process"]["controllable_perturbation"]["sample_times"]}_para_{cfg["process"]["query_paraphrase"]["sampling-params"]["n"]}.{cfg["output_format"]}') for k in cfg["process"]["controllable_perturbation"]["alpha_list"]}
counts = defaultdict(int)

os.makedirs(cfg["target_dir"], exist_ok=True)

if cfg["get_test_set"]:
    test_path = os.path.join(cfg["target_dir"], f'{cfg["data"]["dataset_name"]}_{"|".join(str(item) for item in cfg["process"]["controllable_perturbation"]["alpha_list"])}_{cfg["process"]["controllable_perturbation"]["sample_times"]}_para_{cfg["process"]["query_paraphrase"]["sampling-params"]["n"]}-test.{cfg["output_format"]}')
    test_count = 0
    
if cfg["output_format"] == "jsonl":
    files = {k: open(v, "w") for k, v in paths.items()}
    if cfg["get_test_set"]:
        test_file = open(test_path, "w")

def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", r"\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"\\boxed\{(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) == 0:
        res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 114.514

pattern = r'(?im)^\s*Rephrase .*? question:\s*'
from utils import get_line_count
line_count = get_line_count(input_path)

test_result = []
results = defaultdict(list)
with open(input_path, "r") as in_f:
    for line in tqdm(in_f, total=line_count):
        item = json.loads(line)
        for generated_text in item["generated_texts"][:1]:
            result = {
                "data_source": f"{cfg['data']['dataset_name']}-{item['id']}",
                "prompt": [
                    {"role": "user", "content": re.sub(pattern, '', generated_text)}
                ],
                "reward_model": {
                    "style": "rule",
                    "ground_truth":  str(item["answer"])
                },
                "extra_info": {
                    "id": item["id"],
                    "max_fluct": item["max_fluct"],
                    "code": item["code"]
                }
            }
            if item["system"]:
                result["prompt"].insert(0, {"role": "system", "content": item["system"]})
            if cfg["get_test_set"] and random.random() < cfg["test_ratio"]:
                test_count += 1
                if cfg["output_format"] == "jsonl":
                    test_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                else:
                    test_result.append(result)
            else:
                counts[item["max_fluct"]] += 1
                if cfg["output_format"] == "jsonl":
                    files[item["max_fluct"]].write(json.dumps(result, ensure_ascii=False) + "\n")
                else:
                    results[item["max_fluct"]].append(result)

if cfg["output_format"] == "parquet":
    for k, v in results.items():
        df = pd.DataFrame(v)
        df.to_parquet(paths[k], engine="pyarrow", index=False)
    if cfg["get_test_set"]:
        df = pd.DataFrame(test_result)
        df.to_parquet(test_path, engine="pyarrow", index=False)

elif cfg["output_format"] == "json":
    for k, v in results.items():
        with open(paths[k], "w") as f:
            json.dump(v, f, indent=4, ensure_ascii=False)
    if cfg["get_test_set"]:
        with open(test_path, "w") as f:
            json.dump(test_result, f, indent=4, ensure_ascii=False)
            
elif cfg["output_format"] == "jsonl":
    for v in files.values():
        v.close()
else:
    raise Exception("Error output format")

print()
print("="* 40)
print("Final Results have done!\n")
for k, v in paths.items():
    print()
    print(f"Train set: alpha={k}, sample_times={cfg['process']['controllable_perturbation']['sample_times']}, paraphase_times={cfg['process']['query_paraphrase']['sampling-params']['n']}")
    print(f"Total: {counts[k]}")
    print(f"Output path: {v}")
if cfg["get_test_set"]:
    print()
    print(f"Test set: alpha={'|'.join(str(item) for item in cfg['process']['controllable_perturbation']['alpha_list'])}, sample_times={cfg['process']['controllable_perturbation']['sample_times']}, paraphase_times={cfg['process']['query_paraphrase']['sampling-params']['n']}")
    print(f"Total: {test_count}")
    print(f"Output path: {test_path}")
print("="* 40)
print()