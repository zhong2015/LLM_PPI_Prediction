# coding=utf-8

import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import re
import numpy as np
import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt_preprocess import get_known_entry, get_query_entry
from metrics import RMSE, MAE, eucliDist

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict(sys_prompt, user_prompt, tokenizer, model, pattern):
    # client = OpenAI(base_url="http://chatapi.littlewheat.com/v1",
    #                 api_key="sk-KNAtm2OLQGWGrawWrLerq9V4aDyamaeE5W65UALAJooRdbP0")
    client = OpenAI(base_url="https://api.deepseek.com",
                    api_key="sk-3f42f3d36d634397970c7167863c7e69")
    print("看看有哪些model可以选择:\n", client.models.list())

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
    )

    answer = completion.choices[0].message.content
    # print("answer:\n", answer)
    # https://blog.csdn.net/m0_37738114/article/details/118967886
    pattern = r'[\s\S]*<<Begin>>([\s\S]*)<<End>>[\s\S]*'
    match = re.search(pattern, answer)
    success_flag = False
    success_results = []
    try:
        results = match.group(1)
        print("results:", results)
        result_list = results.strip().split(",")
        for result in result_list:
            try:
                predicted_value = float(result.strip())
            except Exception as e:
                print("LLM fails to predict the [MASK] , since its predicted value is:\n {0}".format(result))
            else:
                if predicted_value > 0 and predicted_value < 1:
                    success_results.append(predicted_value)
                else:
                    print("the predicted value {0} exceeds the range!".format(predicted_value))
                    success_flag = False
                    break
    except Exception as e:
        print("LLM fails to predict the [MASK] , since its answer:\n {0}\n does not match the pattern start with <<Begin>> and end with <<End>>".format(answer))
    else:
        success_flag = True
    return success_flag, success_results

if __name__ == '__main__':
    train_file_path = "C:\\Users\\zhong\\Desktop\\DS\\DS_nonnegatve_CPU Server\\1_zyr_224308_train.txt"
    val_file_path = "C:\\Users\\zhong\\Desktop\\DS\\DS_nonnegatve_CPU Server\\1_zyr_224308_val.txt"
    test_file_path = "C:\\Users\\zhong\\Desktop\\DS\\DS_nonnegatve_CPU Server\\1_zyr_224308_test.txt"
    model_path = 'D:\downloaded_LLM\deepseek-math-7b-rl'
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='right', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

    file_path = "D:\\博士后期间\\稳工\\重要已读的知识\\LLM\\LLM-Codes\\LLM_SHDI\\"
    if os.path.exists((file_path + "fail_queries.txt") and (file_path + "fail_targets.txt") and (
            file_path + "success_results.txt") and (file_path + "success_targets.txt")):
        fail_queries_rfile = open("fail_queries.txt", "r", encoding="utf8")
        fail_targets_rfile = open("fail_targets.txt", "r", encoding="utf8")
        fail_queries_text = fail_queries_rfile.read()
        queries = fail_queries_text.split(",")
        fail_targets_text = fail_targets_rfile.read()
        targets = fail_targets_text.split(",")
        targets = [float(data) for data in targets]
        fail_queries_rfile.close()
        fail_targets_rfile.close()
    else:
        label_count = 50
        query_template = "Input: Row Index={0}, Column Index={1}\n"
        queries, targets = get_query_entry(test_file_path, query_template, label_count=label_count)

    known_count = 180
    sys_prompt = """
    You are  a top-tier mathematician who is exceptionally skilled at predicting missing values of a symmetric sparse matrix.
    """
    user_template = """
    You are a top-tier mathematician who is exceptionally skilled at predicting missing values of a symmetric sparse matrix.
    Note that the values of a symmetric sparse matrix are all decimal numbers ranging between 0 and 1.
    The current task is to predict the decimal value at a specific position in the matrix (identified by its row and column indices) based on the provided examples.
    Known Value Examples:
    """
    known_template = "Input: Row Index={0}, Column Index={1}\n\tAnswer: {2}\n\t"
    known_entries = get_known_entry(train_file_path, known_template, known_count=known_count)
    user_prompt = user_template
    for known_entry in known_entries:
        user_prompt += known_entry
    query_template_base = """
    Now given the following positions of the symmetric matrix:
    {0}
    please reason step by step, and predict their values one by one .
    Finally, separate the predicted values with commas, start with <<Begin>> and end with <<End>>.
    """
    pattern = r'[\s\S]*\$[\s\S]*(\d\.\d+)[\s\S]*\$[\s\S]*'
    user_prompt += query_template_base.format(queries)
    success_flag, results = predict(sys_prompt, user_prompt, tokenizer, model, pattern)

    if os.path.exists((file_path + "fail_queries.txt") and (file_path + "fail_targets.txt") and (
            file_path + "success_results.txt") and (file_path + "success_targets.txt")):
        success_results_rfile = open("success_results.txt", "r", encoding="utf8")
        success_targets_rfile = open("success_targets.txt", "r", encoding="utf8")
        success_results_text = success_results_rfile.read()
        success_results = success_results_text.split(",")
        success_results = [float(data) for data in success_results]
        success_targets_text = success_targets_rfile.read()
        success_targets = success_targets_text.split(",")
        success_targets = [float(data) for data in success_targets]
        success_results_rfile.close()
        success_targets_rfile.close()
        results.extend(success_results)
        targets.extend(success_targets)

    if success_flag:
        print("Warning: len(predict) != len(target)") if len(results) != len(targets) else print("接下来可以进行metric的计算")
        results = np.array(results)
        targets = np.array(targets)
        rmse = RMSE(results, targets)
        print("rmse:", rmse)
        mae = MAE(results, targets)
        print("mae:", mae)
    else:
        print("openAI fails to predict!")



