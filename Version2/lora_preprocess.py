# coding=utf-8

import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, DefaultDataCollator
from lora_config import ProjectConfig

config = ProjectConfig()

def get_known_data(file_path):
    with open(file_path, "rb") as f:
        data = f.read().decode("utf-8")  # 以UTF-8编码读取文件内容
    dataList = data.split("\n")
    dataList = [data.strip() for data in dataList if data != '']
    return dataList

def get_ids(trainfile_path, valfile_path, template, known_template, query_template, label_template, tokenizer,
                train_joint_symbol, inference_joint_symbol, inference_data_num, train_ratio, entry_size):
    train_data_list = get_known_data(trainfile_path)
    val_data_list = get_known_data(valfile_path)
    train_data_list.extend(val_data_list)
    if inference_data_num > 0 and len(train_data_list) > inference_data_num:
        print("用于推理的已知值数目=", inference_data_num)
        inference_data_list = train_data_list[-inference_data_num:]
        train_data_list = train_data_list[:len(train_data_list) - inference_data_num]
        inference_data_wfile = open("inference_data.txt", "w", encoding="utf8")
        inference_data_text = inference_joint_symbol.join(inference_data_list)
        inference_data_wfile.write(inference_data_text)
        inference_data_wfile.close()
    else:
        print("推理时将使用零样本学习！")
    data_num = len(train_data_list)
    print("用于训练和验证的已知值数目=", data_num)
    known_prompt_list = []
    entry_list = []
    label_list = []
    for idx, train_data in enumerate(train_data_list):
        train_meta_data = train_data.split('::')
        train_row = train_meta_data[0]
        train_col = train_meta_data[1]
        train_value = train_meta_data[2]
        known_prompt = known_template.format(train_row, train_col, train_value)
        known_prompt_list.append(known_prompt)
        if (idx + 1) % entry_size == 0:
            entry_id = int((idx + 1) / entry_size)
            val_data = train_data_list[-1 * entry_id]
            val_meta_data = val_data.split('::')
            val_row = val_meta_data[0]
            val_col = val_meta_data[1]
            val_value = val_meta_data[2]
            query_prompt = query_template.format(val_row, val_col)
            entry_prompt = template + train_joint_symbol.join(known_prompt_list) + query_prompt
            entry_list.append(entry_prompt)
            label_list.append(label_template.format(val_value))
            known_prompt_list = []
            if data_num - (idx + 1 + entry_id) < entry_size:
                break
    input_ids = [tokenizer.encode(text=entry, text_pair=label) for entry, label in zip(entry_list, label_list)]
    train_sample_num = int(len(input_ids) * train_ratio)
    print("用于训练和验证的entry数目={0}, 其中，用于训练的entry数目={1}，而用于验证的entry数目={2}".format(len(input_ids),
                                                                                                         train_sample_num,
                                                                                                         len(input_ids) - train_sample_num))
    train_sample_ids = {'input_ids': input_ids[:train_sample_num]}
    val_sample_ids = {'input_ids': input_ids[train_sample_num:]}
    return train_sample_ids, val_sample_ids

def collate_fn(batch):
    batch_size = len(batch)
    tokenizer = AutoTokenizer.from_pretrained(config.pre_model, padding_side='right', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    labels = []
    for idx, sample in enumerate(batch):
        sample_ids = sample['input_ids']
        begin_indices = [idx for idx, word_id in enumerate(sample_ids) if word_id == tokenizer.convert_tokens_to_ids(tokenizer.bos_token)]
        entry_length = begin_indices[1] + 1
        """
        由于这里所使用的model是一个LlamaForCausalLM类对象，所以根据如下网址可知，将一个样本所对应label中某个位置上的索引值设为-100后，
        那么该位置上的真实值与对应的预测值不参与损失值的计算。
        https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/llama#transformers.LlamaForCausalLM
        """
        label_ids = [-100] * entry_length + sample_ids[entry_length:]
        labels.append(label_ids)
    if batch_size == 1:
        data_collator = DefaultDataCollator(return_tensors='pt')
        output_batch = data_collator(batch)
    else:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors='pt')
        output_batch = data_collator(batch)
    batch_max_length = output_batch['input_ids'].shape[1]
    final_labels = []
    for label in labels:
        padding_length = batch_max_length - len(label)
        final_label = label + [-100] * padding_length
        final_labels.append(final_label)
    output_batch['labels'] = torch.Tensor(final_labels)
    return output_batch

def get_dataloader(trainfile_path, valfile_path, template, known_template, query_template, label_template, tokenizer,
                   train_joint_symbol, inference_joint_symbol, inference_data_num, train_ratio, entry_size, batch_size):
    train_sample_ids, val_sample_ids = get_ids(trainfile_path, valfile_path, template, known_template, query_template,
                                               label_template, tokenizer, train_joint_symbol, inference_joint_symbol,
                                               inference_data_num, train_ratio, entry_size)
    train_dataset = Dataset.from_dict(train_sample_ids)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  drop_last=True)
    val_dataset = Dataset.from_dict(val_sample_ids)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=collate_fn,
                                drop_last=True)
    return train_dataloader, val_dataloader

def get_mean_known_value(trainfile_path, valfile_path):
    known_value_list = []
    train_data_list = get_known_data(trainfile_path)
    val_data_list = get_known_data(valfile_path)
    train_data_list.extend(val_data_list)
    for idx, train_data in enumerate(train_data_list):
        train_meta_data = train_data.split('::')
        train_value = train_meta_data[2]
        known_value_list.append(float(train_value))
    return np.mean(known_value_list)

if __name__ == '__main__':
    template = """
    You are a top-tier mathematician who is exceptionally skilled at predicting missing values of a symmetric sparse matrix.
    Note that the values of a symmetric sparse matrix are all decimal numbers ranging between 0 and 1.
    The current task is to predict the decimal value at a specific position in the matrix (identified by its row and column indices) based on the provided examples.
    Examples:
    """
    known_template = "Input: Row Index={0}, Column Index={1}\nAnswer: ${2}$\n"
    query_template = "Input: Row Index={0}, Column Index={1}\nAnswer: "
    label_template = "${0}$"
    """
    此时的tokenizer是一个LlamaTokenizerFast类对象，其print输出如下所示：
    LlamaTokenizerFast(name_or_path='D:\downloaded_LLM\deepseek-math-7b-rl', vocab_size=100000, model_max_length=4096, 
                       is_fast=True, padding_side='left', truncation_side='right', 
                       special_tokens={'bos_token': '<｜begin▁of▁sentence｜>', 'eos_token': '<｜end▁of▁sentence｜>'}, 
                                       clean_up_tokenization_spaces=False),  
                                       added_tokens_decoder={
	                                        100000: AddedToken("<｜begin▁of▁sentence｜>", rstrip=False, lstrip=False, 
	                                                           single_word=False, normalized=True, special=True),
	                                        100001: AddedToken("<｜end▁of▁sentence｜>", rstrip=False, lstrip=False, 
	                                                           single_word=False, normalized=True, special=True),}
    """
    tokenizer = AutoTokenizer.from_pretrained(config.pre_model, padding_side='right', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataloader, val_dataloader = get_dataloader(config.trainfile_path, config.valfile_path, template,
                                                      known_template, query_template, label_template, tokenizer,
                                                      config.train_joint_symbol, config.inference_joint_symbol,
                                                      config.inference_data_num, config.train_ratio, config.entry_size,
                                                      config.batch_size)
