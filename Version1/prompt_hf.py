# coding=utf-8

import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt_preprocess import get_known_entry, get_query_entry
from metrics import RMSE, MAE, eucliDist

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict(prompt, query, tokenizer, model, pattern):
    # tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt + query}], add_generation_prompt=True,
                                           tokenize=True, return_tensors="pt", return_dict=True).to(device)

    print("prompt token count:", prompt['input_ids'].shape[1])
    gen_kwargs = {"max_new_tokens": 1024,
                  "pad_token_id": tokenizer.eos_token_id}
    with torch.no_grad():
        outputs = model.generate(**prompt, **gen_kwargs)
    generated_text = outputs[:, prompt['input_ids'].shape[1]:-1]
    match = re.search(pattern, tokenizer.decode(generated_text[0]))
    result = 0
    success_flag = False
    try:
        result = match.group(1)
        print("result:", result)
        result = float(result.strip())
    except Exception as e:
        print("=" * 80)
        print("LLM fails to predict, since its answer is:\n{0}".format(tokenizer.decode(generated_text[0])))
        print("=" * 80)
    else:
        success_flag = True
    return success_flag, result

if __name__ == '__main__':
    train_file_path = "C:\\Users\\zhong\\Desktop\\DS\\DS_nonnegatve_CPU Server\\1_zyr_224308_train.txt"
    val_file_path = "C:\\Users\\zhong\\Desktop\\DS\\DS_nonnegatve_CPU Server\\1_zyr_224308_val.txt"
    test_file_path = "C:\\Users\\zhong\\Desktop\\DS\\DS_nonnegatve_CPU Server\\1_zyr_224308_test.txt"
    model_path = 'D:\downloaded_LLM\deepseek-math-7b-instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='right', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

    file_path = "D:\\博士后期间\\稳工\\重要已读的知识\\LLM\\LLM-Codes\\LLM_SHDI\\"
    if os.path.exists((file_path + "fail_queries.txt") and (file_path + "fail_targets.txt") and (
            file_path + "success_results.txt") and (file_path + "success_targets.txt")):
        fail_queries_rfile = open("fail_queries.txt", "r", encoding="utf8")
        fail_targets_rfile = open("fail_targets.txt", "r", encoding="utf8")
        success_results_rfile = open("success_results.txt", "r", encoding="utf8")
        success_targets_rfile = open("success_targets.txt", "r", encoding="utf8")
        fail_queries_text = fail_queries_rfile.read()
        queries = fail_queries_text.split(",")
        fail_targets_text = fail_targets_rfile.read()
        targets = fail_targets_text.split(",")
        targets = [float(data) for data in targets]
        success_results_text = success_results_rfile.read()
        success_results = success_results_text.split(",")
        success_results = [float(data) for data in success_results]
        success_targets_text = success_targets_rfile.read()
        success_targets = success_targets_text.split(",")
        success_targets = [float(data) for data in success_targets]
        fail_queries_rfile.close()
        fail_targets_rfile.close()
        success_results_rfile.close()
        success_targets_rfile.close()
        fail_queries = []
        fail_targets = []
    else:
        label_count = 50

        query_template = "Input: Row Index={0}, Column Index={1}"
        # query_template = "{0}::{1}"
        # query_template = "Input: {0}::{1}"

        queries, targets = get_query_entry(test_file_path, query_template, label_count=label_count)
        success_results = []
        success_targets = []
        fail_queries = []
        fail_targets = []

    known_count = 180

    template = """
    You are a top-tier mathematician who is exceptionally skilled at predicting missing values of a symmetric sparse matrix.
    Note that the known values of a symmetric sparse matrix are all decimal numbers ranging between 0 and 1.
    The current task is to predict the decimal value at a specific position in the matrix (identified by its row and column indices) based on the provided examples.
    Hence, known value examples are given as follows:
    """
    # template = """
    # Given the known values of a symmetric sparse matrix, they are all decimal numbers ranging between 0 and 1.
    # Note that the format of the known values is structured as row::col$value$. For example, 10::12$0.5$ indicates
    # that the value at the 10th row and 12th column in the symmetric matrix is 0.5.
    # Hence, known value examples are given as follows:
    # """
    # template = """
    # Given the known values of a symmetric sparse matrix, they are all decimal numbers ranging between 0 and 1.
    # Note that the format of the known values is structured as:
    # Input: row::col\nAnswer: $value$.
    # For example,
    # Input: 10::12\nAnswer: $0.5$
    # This example indicates that the value at the 10th row and 12th column in the symmetric matrix is 0.5.
    # Hence, known value examples are given as follows:
    # """

    known_template = "Input: Row Index={0}, Column Index={1}\nAnswer: ${2}$\n"
    # known_template = "{0}::{1}${2}$\n"
    # known_template = "Input: {0}::{1}\nAnswer: ${2}$\n"

    known_entries = get_known_entry(train_file_path, known_template, known_count=known_count)
    known_prompt = "".join(known_entries)
    prompt = template + known_prompt
    query_template_base = """
    Based on above known values, 
    {0}
    Please predict its value, and put your predicted value between two $ symbols. 
    The answer is:
    """
    pattern = r'[\s\S]*\$[\s\S]*(\d\.\d+)[\s\S]*\$[\s\S]*'
    for query, target in zip(queries, targets):
        new_query = query_template_base.format(query)
        success_flag, result = predict(prompt, new_query, tokenizer, model, pattern)
        if success_flag and result > 0 and result < 1:
            success_results.append(result)
            success_targets.append(target)
        else:
            fail_queries.append(query)
            fail_targets.append(target)

    if len(fail_queries) > 0:
        fail_queries_wfile = open("fail_queries.txt", "w", encoding="utf8")
        fail_targets_wfile = open("fail_targets.txt", "w", encoding="utf8")
        success_results_wfile = open("success_results.txt", "w", encoding="utf8")
        success_targets_wfile = open("success_targets.txt", "w", encoding="utf8")
        fail_queries_str = ",".join(fail_queries)
        fail_queries_wfile.write(fail_queries_str)
        fail_targets = [str(data) for data in fail_targets]
        fail_targets_str = ",".join(fail_targets)
        fail_targets_wfile.write(fail_targets_str)
        success_results = [str(data) for data in success_results]
        success_results_str = ",".join(success_results)
        success_results_wfile.write(success_results_str)
        success_targets = [str(data) for data in success_targets]
        success_targets_str = ",".join(success_targets)
        success_targets_wfile.write(success_targets_str)
        fail_queries_wfile.close()
        fail_targets_wfile.close()
        success_results_wfile.close()
        success_targets_wfile.close()
    else:
        print("Warning: len(success_results) != len(success_targets), since the predicts are:\n{0}".format(
            success_results)) if len(success_results) != len(success_targets) else print("接下来可以进行metric的计算")
        success_results = np.array(success_results)
        success_targets = np.array(success_targets)
        rmse = RMSE(success_results, success_targets)
        print("rmse:", rmse)
        mae = MAE(success_results, success_targets)
        print("mae:", mae)



