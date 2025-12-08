
import contextlib
from io import StringIO
import re
import subprocess
import json
from tqdm import tqdm
import os
from contextlib import contextmanager
import signal
import threading
import yaml

with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)
    
from utils import set_seed
set_seed(cfg["process"]["seed"])

input_path = os.path.join(cfg["process"]["tmp_folder"], "template_and_code_generation", f"{cfg['data']['dataset_name']}_generated.jsonl")
output_path = os.path.join(cfg["process"]["tmp_folder"], "template_and_code_generation", f"{cfg['data']['dataset_name']}_parsed.jsonl")

success = 0
template_generation_fault_count = 0 # 无法生成模板
template_generation_mistake_count = 0 # 生成的模板与原始问题不匹配
python_generation_fault_count = 0 # 无法生成 Python 代码
python_run_fault_count = 0 # 无法运行的 Python 代码
python_run_mistake_count = 0 # Python 代码运行结果错误
template_python_not_algined_count = 0 # 模板中的变量与 Python 代码中的变量不对齐

# @contextmanager
# def time_limit(seconds):
#     def signal_handler(signum, frame):
#         raise Exception("Timed out!")
#     signal.signal(signal.SIGALRM, signal_handler)
#     signal.alarm(seconds)
#     try:
#         yield
#     finally:
#         signal.alarm(0)

# def runcode(code):
#     """执行 Python 代码并返回输出"""
#     output = ""
#     try:
#         with time_limit(1):
#             # 捕获标准输出
#             with StringIO() as buf, contextlib.redirect_stdout(buf):
#                 exec(code, {}, {})
#                 output = buf.getvalue().strip()
#     except Exception as e:
#         return None
    
#     try:
#         return float(output)
#     except ValueError:
#         if re.search(r'\d', output):
#              output = None
#         else:
#             return output

import multiprocessing
from queue import Empty
import itertools


class SafeExecutor:
    def __init__(self):
        self._task_q = multiprocessing.Queue()
        self._result_q = multiprocessing.Queue()
        self._req_id_iter = itertools.count()
        self._start_worker()

    def _start_worker(self):
        """启动一个常驻子进程，负责执行代码"""
        self._proc = multiprocessing.Process(
            target=self._worker,
            args=(self._task_q, self._result_q),
        )
        self._proc.daemon = True
        self._proc.start()

    @staticmethod
    def _worker(task_q, result_q):
        
        def mock_input(prompt=""):
            return "" 

        import builtins as _builtins
        _builtins.input = mock_input
        _builtins.quit = lambda *a, **kw: ""
        _builtins.exit = lambda *a, **kw: ""

        while True:
            item = task_q.get()
            if item is None:
                break  # 收到结束信号
            req_id, code = item
            try:
                with StringIO() as buf, contextlib.redirect_stdout(buf):
                    exec(code, {}, {})
                    output = buf.getvalue().strip()
                result_q.put((req_id, "ok", output))
            except Exception as e:
                result_q.put((req_id, "error", repr(e)))

    def _restart_worker(self):
        """在超时或崩溃时重启子进程"""
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join()
        self._start_worker()

    def runcode(self, code, timeout=2):
        req_id = next(self._req_id_iter)
        self._task_q.put((req_id, code))

        try:
            while True:
                rid, status, payload = self._result_q.get(timeout=timeout)
                if rid != req_id:
                    continue
                break
        except Empty:
            self._restart_worker()
            return None

        if status == "error":
            return None

        output = payload

        try:
            return float(output)
        except ValueError:
            if re.search(r'\d', output):
                return None
            else:
                return output

    def close(self):
        """优雅关闭子进程（非必须，但你可以在程序结束前调用）"""
        try:
            self._task_q.put(None)
        except:
            pass
        if self._proc.is_alive():
            self._proc.join()

# 全局复用同一个执行器
_executor = None

def get_executor():
    global _executor
    if _executor is None:
        _executor = SafeExecutor()
    return _executor

def runcode(code, timeout=1):
    executor = get_executor()
    return executor.runcode(code, timeout=timeout)

def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", r"\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"\\boxed\{(\d+(\.\d+)?)", text)  # 匹配 \\boxed
    if len(res) == 0:
        res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 114.514 + random.random() * 10

c = 0

def extract_content(item):
    
    global success, python_run_fault_count, python_run_mistake_count, python_generation_fault_count, template_generation_fault_count, template_generation_mistake_count, template_python_not_algined_count
    
    generation = item["generated_texts"][0]
    # print(generation)
    # 提取第一次出现的### template后的内容
    template_match = re.search(r'### (?:Query|Query Template|Template):(.*?)(?=###|$)', generation, re.DOTALL | re.IGNORECASE)
    template_content = template_match.group(1).strip() if template_match else None
    
    # 提取### python代码块中的代码
    python_code_match = re.search(r'### Python Code:\s*```(?:python)?\s*(.*?)\s*```', generation, re.DOTALL | re.IGNORECASE)
    python_code = python_code_match.group(1).strip() if python_code_match else None
    
    if python_code is None:
        python_generation_fault_count += 1
    elif template_content is None:
        template_generation_fault_count += 1 
    elif abs(template_content.count(' ') - item["query"].count(' ')) > 0.5 * min(item["query"].count(' '), template_content.count(' ')):
        template_generation_mistake_count += 1
    else:
        python_result = runcode(python_code)
        if python_result is None:
            global c
            python_run_fault_count += 1
        elif isinstance(python_result, str):
            item["answer"] = python_result
        elif abs(python_result - extract_last_num(str(item["answer"]))) > 1e-2:
            python_run_mistake_count += 1
        else:
            variables = re.findall(r'<([^>]+?)>', template_content)
            for var in variables:
                pattern = r'\b' + re.escape(var) + r'\s*?='
                if re.search(pattern, python_code) is None:
                    template_python_not_algined_count += 1
                    return False
            else:
                success += 1
                item["template"] = template_content
                item["python"] = python_code
            return True
        
    return False

from utils import get_line_count
line_count = get_line_count(input_path)
with open(input_path, "r") as in_f, open(output_path, "w") as out_f:
    for line in tqdm(in_f, total=line_count):
        item = json.loads(line)
        if extract_content(item):
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

print()
print("="* 40)
print("Results for template and code generation have been parsed!\n")
print(f"Success rate: {success} / {line_count} = {success / line_count}")
print(f"The error from template generation: {template_generation_fault_count}")
print(f"The error from code generation: {python_generation_fault_count}")
print(f"The error from a mismatch between the template and the query: {template_generation_mistake_count}")
print(f"The error from the runtime error: {python_run_fault_count}")
print(f"The error from code running with incorrect results: {python_run_mistake_count}")
print(f"The error from a mismatch between the query and the code: {template_python_not_algined_count}")
print("="* 40)