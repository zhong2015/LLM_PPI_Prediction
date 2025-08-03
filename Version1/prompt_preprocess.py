# coding=utf-8

def get_known_data(file_path):
    with open(file_path, "rb") as f:
        data = f.read().decode("utf-8")  # 以UTF-8编码读取文件内容
    dataList = data.split("\n")
    dataList = [data.strip() for data in dataList if data != '']
    return dataList

def get_known_entry(file_path, known_template, known_count=0):
    data_list = get_known_data(file_path)
    known_entry_list = []
    if known_count > 0:
        data_list = data_list[:known_count]
    for data in data_list:
        meta_data = data.split('::')
        row = meta_data[0]
        col = meta_data[1]
        value = meta_data[2]
        known_entry = known_template.format(row, col, value)
        known_entry_list.append(known_entry)
    return known_entry_list

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