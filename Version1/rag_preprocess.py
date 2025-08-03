# coding=utf-8

import hashlib
from haystack import Document

def get_known_data(file_path):
    with open(file_path, "rb") as f:
        data = f.read().decode("utf-8")  
    dataList = data.split("\n")
    dataList = [data.strip() for data in dataList if data != '']
    return dataList

def build_docs(file_path, known_template, joint_symbol="", batch_size=60, entry_num=0):
    docs = []
    data_list = get_known_data(file_path)
    if entry_num > 0:
        data_list = data_list[:entry_num]
    entry_list = []
    for i, data in enumerate(data_list):
        meta_data = data.split('::')
        row = meta_data[0]
        col = meta_data[1]
        value = meta_data[2]
        entry = known_template.format(row, col, value)
        entry_list.append(entry)
        if (i+1) % batch_size == 0:
            sample = joint_symbol.join(entry_list)
            hash_object = hashlib.sha256(sample.encode())
            doc_id = hash_object.hexdigest()[:10]
            doc = Document(id=doc_id, content=sample)
            docs.append(doc)
            entry_list = []
    if len(entry_list) > 0:
        sample = ""
        for entry in entry_list:
            sample += entry
        hash_object = hashlib.sha256(sample.encode())
        doc_id = hash_object.hexdigest()[:10]
        doc = Document(id=doc_id, content=sample)
        docs.append(doc)
    print("Collected {0} documents！".format(len(docs)))
    return docs

def get_query_entry(file_path, query_template, label_count=0):
    data_list = get_known_data(file_path)
    query_entry_list = []
    target_list = []
    if label_count > 0:
        data_list = data_list[:label_count]
    for data in data_list:
        meta_data = data.split('::')
        row = meta_data[0]
        col = meta_data[1]
        value = meta_data[2]
        query_entry = query_template.format(row, col)
        query_entry_list.append(query_entry)
        target_list.append(float(value))
    return query_entry_list, target_list
