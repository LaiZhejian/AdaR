import sys
import json
import os
import re
import random
from tqdm.contrib.concurrent import process_map
from functools import partial
from fractions import Fraction
import multiprocessing
from queue import Empty
import time
import itertools
import yaml

with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)
    
from utils import set_seed
set_seed(cfg["process"]["seed"])

max_flucts=cfg["process"]["controllable_perturbation"]["alpha_list"]
sample_times=cfg["process"]["controllable_perturbation"]["sample_times"]
input_path = os.path.join(cfg["process"]["tmp_folder"], "template_and_code_generation", f"{cfg['data']['dataset_name']}_parsed.jsonl")
output_dir = os.path.join(cfg["process"]["tmp_folder"], "controllable_generation")
output_path =  os.path.join(output_dir, f'{cfg["data"]["dataset_name"]}_{"|".join(str(item) for item in max_flucts)}_{sample_times}.jsonl')

os.makedirs(output_dir, exist_ok=True)

def mock_input(prompt=""):
    return "" 

def extract_last_num(text) -> float:
    if isinstance(text, str):
        text = re.sub(r"(\d),(\d)", r"\g<1>\g<2>", text)  # 处理形如 123,456
        res = re.findall(r"\\boxed\{(\d+(\.\d+)?)", text)  # 匹配 \\boxed
        if len(res) == 0:
            res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
        if len(res) > 0:
            num_str = res[-1][0]
            return float(num_str)
        else:
            return 0.0
    else:
        return text

def check_validity(value, old_value):
    try:
        if isinstance(old_value, str):
            return old_value.strip().lower() == value.strip().lower()
        
        old_value = int(old_value) if float(old_value).is_integer() else float(old_value)
        value = float(value)
        if isinstance(old_value, int):
            if value.is_integer():
                value = int(value)
            else:
                return False
        return value * old_value >= 0
    except ValueError:
        return False

# @timeout(cfg["process"]["controllable_perturbation"]["timeout_seconds_per_sample"])
# def python_run(code):
#     local_env = {
#         '__result__': "",  # 存储 print 的内容
#     }
#     def custom_print(*args, **kwargs):
#     # 将所有参数转换为字符串并拼接
#         sep = kwargs.get('sep', ' ')
#         end = kwargs.get('end', '\n')
#         output = sep.join(str(arg) for arg in args) + end
#         local_env['__result__'] += output
#     local_env['print'] = custom_print
#     exec(code, local_env)
#     return local_env["__result__"]

class SafeExecutor:
    def __init__(self, mock_input_func):
        self.mock_input = mock_input_func
        self._task_q = multiprocessing.Queue()
        self._result_q = multiprocessing.Queue()
        self._req_id_iter = itertools.count()
        self._start_worker()

    def _start_worker(self):
        self._proc = multiprocessing.Process(
            target=self._worker,
            args=(self._task_q, self._result_q, self.mock_input),
        )
        self._proc.daemon = True
        self._proc.start()

    @staticmethod
    def _worker(task_q, result_q, mock_input_func):
        """
        子进程：设置 builtins + 自定义 print + exec 代码，返回结果字符串
        """
        import builtins as _builtins

        # 覆写 builtins

        import builtins as _builtins
        _builtins.input = mock_input_func
        _builtins.quit = lambda *a, **kw: ""
        _builtins.exit = lambda *a, **kw: ""

        while True:
            item = task_q.get()
            if item is None:
                break

            req_id, code = item

            local_env = {'__result__': ""}

            def custom_print(*args, **kwargs):
                sep = kwargs.get('sep', ' ')
                end = kwargs.get('end', '\n')
                output = sep.join(str(arg) for arg in args) + end
                local_env['__result__'] += output

            local_env['print'] = custom_print

            try:
                exec(code, local_env)
                result_q.put((req_id, "ok", local_env["__result__"]))
            except Exception as e:
                result_q.put((req_id, "error", repr(e)))

    def _restart_worker(self):
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join()
        self._start_worker()

    def run(self, code: str, timeout_seconds: float) -> str:
        req_id = next(self._req_id_iter)
        self._task_q.put((req_id, code))

        start = time.time()
        while True:
            remaining = timeout_seconds - (time.time() - start)
            if remaining <= 0:
                self._restart_worker()
                raise TimeoutError("python_run timed out")

            try:
                rid, status, payload = self._result_q.get(timeout=remaining)
            except Empty:
                self._restart_worker()
                raise TimeoutError("python_run timed out (queue empty)")

            if rid != req_id:
                continue

            if status == "error":
                raise RuntimeError(f"Code execution error: {payload}")
            return payload

    def close(self):
        try:
            self._task_q.put(None)
        except:
            pass
        if self._proc.is_alive():
            self._proc.join()


# 全局复用一个 Executor
_executor = None

def get_executor():
    global _executor
    if _executor is None:
        _executor = SafeExecutor(mock_input)
    return _executor


def python_run(code):
    executor = get_executor()
    timeout_seconds = cfg["process"]["controllable_perturbation"]["timeout_seconds_per_sample"]
    return executor.run(code, timeout_seconds)

def randomize_value(original_value, max_fluct=1.0, upper_bound=10**9):
    """
    Returns a value randomly fluctuated within ±(max_fluct * original_value).
    If original_value is int, the result is rounded back to int.
    """

    lower_bound = original_value * (1 - max_fluct)
    if original_value > 0:
        lower_bound = max(1 if isinstance(original_value, int) else 0.01, lower_bound)
    upper_bound = min(original_value * (1 + max_fluct), upper_bound)
    if original_value < 0:
        upper_bound = min(-1 if isinstance(original_value, int) else -0.01, upper_bound)
    if isinstance(original_value, float) and 0 < original_value < 1:
        lower_bound = max(0.01, lower_bound)
        upper_bound = min(0.99, upper_bound)

    if random.random() < 0.5 and original_value != 0:
        for _ in range(cfg["process"]["controllable_perturbation"]["retry_times_inner"]):
            new_value = random.uniform(lower_bound, upper_bound)
            
            if isinstance(original_value, int):
                new_value =  int(round(new_value))
            else:
                new_value = round(new_value, 2)
                
            if new_value != original_value:
                break
            
            max_fluct += 0.1
            lower_bound = original_value * (1 - max_fluct)
            if original_value > 0:
                lower_bound = max(1 if isinstance(original_value, int) else 0.01, lower_bound)
            upper_bound = original_value * (1 + max_fluct)
            if original_value < 0:
                upper_bound = min(-1 if isinstance(original_value, int) else -0.01, upper_bound)
            if isinstance(original_value, float) and 0 < original_value < 1:
                lower_bound = max(0.01, lower_bound)
                upper_bound = min(0.99, upper_bound)
    else:
        new_value = random.uniform(lower_bound, upper_bound)
        
        if isinstance(original_value, int):
            new_value =  int(round(new_value))
        else:
            new_value = round(new_value, 2)
        
    return new_value


def randomize_code(max_fluct: float, original_code: str, original_query=None, original_ans=None):
    # Split the code by lines
    lines = original_code.split('\n')

    # We’ll collect lines until we hit the first consecutive blank line
    # (i.e., an empty line).
    variable_lines = []

    for i, line in enumerate(lines):
        if variable_lines == [] and line.strip() == "":
            continue
        # Detect if the line is empty
        if line.strip() == "":
            # This is the first consecutive newline => stop collecting variable lines
            break
        else:
            variable_lines.append(line)
    
    # Use a regex to match lines of the form: name = number
    
    pattern = re.compile(r'^(\s*\w+)\s*=\s*(\d[\d/ ]*|\d*\.\d*)\s*(#.*)?$')
    pattern2 = re.compile(r'^(\s*\w+)\s*=\s*(.*?)\s*(#.*)?$')
    
    # Filter the template not aligned sample
    for line in variable_lines:
        match = pattern.match(line)
        if match:
            prefix = match.group(1)
            if f"<{prefix}>" not in original_query:
                return {"template_python_not_algined_count": None}
            
    variable_num = 0
    for line in variable_lines:
        match = pattern.match(line)
        if match:
            variable_num += 1
    

    for variable_limits in range(variable_num, 0, -1):
        for _ in range(cfg["process"]["controllable_perturbation"]["retry_times_inner"]):
            variable_count = 0
            new_variable_lines = []
            replaced_variables = []
            for idx, line in enumerate(variable_lines):
                match = pattern.match(line)
                if match:
                    # Extract the variable name, the original numeric value, and any trailing spaces
                    prefix = match.group(1)  # e.g. "variables_a"
                    original_value_str = match.group(2)  # e.g. "150"
                    suffix = match.group(3)  # trailing spaces if any
                    suffix = suffix if suffix else ""
                    
                    if variable_limits == variable_count:
                        new_value_str = original_value_str
                    else:
                        # Determine if it’s int or float
                        if '/' in original_value_str:
                            original_value_str = original_value_str.replace('//', '/')
                            original_value_str = original_value_str.replace(' ', '')
                            try:
                                original_value = Fraction(original_value_str)
                            except:
                                raise Exception(line)
                            numerator = original_value.numerator    # 分子
                            denominator = original_value.denominator  # 分母
                            if numerator == 1:
                                denominator = randomize_value(denominator, max_fluct=max_fluct)
                            else:
                                numerator = randomize_value(numerator, max_fluct=max_fluct)
                            new_value_str = f"{numerator}/{denominator}"
                        else:
                            if abs(float(original_value_str) - int(float(original_value_str))) < 1e-6:
                                original_value = int(float(original_value_str))
                            else:
                                original_value = float(original_value_str)
                            new_val = randomize_value(original_value, max_fluct=max_fluct, upper_bound=100 if 'percentage' in prefix else 10 ** 9)
                            new_value_str = str(new_val)
                            
                    replaced_variables.append((prefix, str(new_value_str)))
                    # Rebuild the line
                    new_variable_lines.append(prefix + " = " + new_value_str + suffix)
                    variable_count += 1
                else:
                    new_variable_lines.append(line)
                
                match = pattern2.match(line)
                if match:
                    prefix = match.group(1)  # e.g. "variables_a"
                    original_value_str = match.group(2)  # e.g. "150"
                    suffix = match.group(3)  # trailing spaces if any
                    suffix = suffix if suffix else ""

                    replaced_variables.append((prefix, str(original_value_str)))
                    
            final_code = '\n'.join(new_variable_lines) + '\n' + '\n'.join(lines[i:])
            try:
                result = python_run(final_code)
                if check_validity(result, original_ans):
                    if original_query:
                        for var, new_value in replaced_variables:
                            original_query = original_query.replace(f"<{var}>", str(new_value))
                        return {
                            "new_query": original_query, 
                            "new_code": final_code, 
                            "new_ans": float(result)
                            }
                    else:
                        return final_code
                else:
                    # raise exception("The code is not valid.")
                    pass
            except TimeoutError as e:
                raise e
            except Exception as e:
                pass

# @timeout(cfg["process"]["controllable_perturbation"]["timeout_seconds_total"])
# def randomize_code_multiple_times(times, group_r, *args, **kwargs):
#     for _ in range(times):
#         r = randomize_code(*args, **kwargs)
#         if 'new_ans' in r:
#             # messages = [
#             #     {"role": "system", "content": "Below is an instruction that describes a task. Write a response that appropriately completes the request. Output each step in a separate line, and explicitly state the final answer after the final step within \\boxed{}."},
#             #     {"role": "user", "content": r["new_query"]} # type: ignore
#             # ]

#             # r["prompt"] = tokenizer.apply_chat_template(
#             #     messages,
#             #     tokenize=False,
#             #     add_generation_prompt=True,
#             #     enable_thinking=False)

#             # 将r的内容均添加到item中
#             r["max_fluct"] = kwargs["max_fluct"]
#             group_r.append(r)

def randomize_code_multiple_times(times, group_r, *args, **kwargs):
    total_timeout = cfg["process"]["controllable_perturbation"]["timeout_seconds_total"]
    start_total = time.time()

    for _ in range(times):
        # 总时间检查
        if time.time() - start_total > total_timeout:
            # 超出总时间直接结束
            break

        try:
            r = randomize_code(*args, **kwargs)
        except TimeoutError:
            # 单次超时，略过这个样本，继续下一个
            continue

        if r and 'new_ans' in r:
            r["max_fluct"] = kwargs["max_fluct"]
            group_r.append(r)


def process_item(item, others):
    
    perturbed_results = []
    for max_fluct in max_flucts:
        try:
            randomize_code_multiple_times(times=sample_times // len(max_flucts),
                                        group_r=perturbed_results,
                                        max_fluct=max_fluct, 
                                        original_code=item["python"],
                                        original_query=item["template"],
                                        original_ans=extract_last_num(item["answer"]))
        except:
            pass
    
    if perturbed_results:
        item.update({"perturbed": perturbed_results})
    else:
        item.update({"random_generate_difficult_count": None})
    return item

data = []
with open(input_path, "r") as f:
    for line in f:
        data.append(json.loads(line))

process_func = partial(process_item, others=None)
max_workers = cfg["process"]["controllable_perturbation"]["max_workers"]
print(type(max_workers))
if max_workers == -1:
    max_workers=os.cpu_count() // 2

process_results = process_map(
    process_func,
    data,
    max_workers=max_workers,
    chunksize=cfg["process"]["controllable_perturbation"]["chunk_size"],
)


results = []
template_python_not_algined_count = []
random_generate_difficult_count = []
for item in process_results:
    if "perturbed" in item:
        results.append(item)
    else:
        if "template_python_not_algined_count" in item:
            template_python_not_algined_count.append(item.get("id", item.get("idx", None)))
        elif "random_generate_difficult_count" in item:
            random_generate_difficult_count.append(item.get("id", item.get("idx", None)))

with open(output_path, "w") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print()
print("="* 40)
print("Parsed variable sets have been perturbed!\n")
print(f"Success rate：{len(results)} / {len(data)} = {len(results) / len(data) * 100:.2f}%")
print(f"The error from a mismatch between the code and the query: {len(template_python_not_algined_count)}, e.g. id: {template_python_not_algined_count[:5]}")
print(f"The number of unsuccessful perturbations: {len(random_generate_difficult_count)}, e.g. id: {random_generate_difficult_count[:5]}")
print("="* 40)
print()