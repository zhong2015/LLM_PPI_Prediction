# coding=utf-8

import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from rag_preprocess import build_docs, get_query_entry
from retrieval_pipeline import BM25_pipeline, embedding_pipeline, BM25_embedding_pipeline
from metrics import RMSE, MAE, eucliDist

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict(document_store, query, template, tokenizer, model, pattern, embedding_flag=True, multi_flag=True, n_retrieves=3):
    if multi_flag:
        prompt = BM25_embedding_pipeline(document_store, query, template, n_retrieves)
    else:
        if embedding_flag:
            prompt = embedding_pipeline(document_store, query, template, n_retrieves)
        else:
            prompt = BM25_pipeline(document_store, query, template, n_retrieves)

    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                           tokenize=True, return_tensors="pt", return_dict=True).to(device)
    print("prompt token count:", prompt['input_ids'].shape[1])
    gen_kwargs = {"max_new_tokens": 1024,
                  "pad_token_id": tokenizer.eos_token_id,
                  "do_sample": True}
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
    model_path = 'D:\downloaded_LLM\deepseek-math-7b-rl'
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='right', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

    file_path = "D:\\LLM\\LLM-Codes\\LLM_SHDI\\"
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
        queries, targets = get_query_entry(test_file_path, query_template, label_count=label_count)
        success_results = []
        success_targets = []
        fail_queries = []
        fail_targets = []

    known_template = "Input: Row Index={0}, Column Index={1}\n\tAnswer: ${2}$\n\t"
    docs = build_docs(train_file_path, known_template, joint_symbol="", batch_size=60, entry_num=72000)
    # document_store = ElasticsearchDocumentStore(hosts="http://localhost:9200")
    document_store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    if document_store.count_documents() == 0:
        print("document_store is empty")
    else:
        print("document count:", document_store.count_documents())
        print("deleting all documents...")
        existing_docs = document_store.filter_documents()
        ids = []
        for doc in existing_docs:
            ids.append(doc.id)
        document_store.delete_documents(ids)
        assert document_store.count_documents() == 0, "document_store is not empty！"
        print("Deleted！document_store is empty！")

    n_retrieves = 3
    multi_flag = True
    embedding_flag = False
    print("sample num for training={0}, sample num for testing={1}, multi_flag={2}, embedding_flag={3}".format(
        n_retrieves * 60, len(targets), multi_flag, embedding_flag))

    if embedding_flag or multi_flag:
        doc_writer = DocumentWriter(document_store=document_store)
        doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        indexing_pipe = Pipeline()
        indexing_pipe.add_component(instance=doc_embedder, name="doc_embedder")
        indexing_pipe.add_component(instance=doc_writer, name="doc_writer")
        indexing_pipe.connect("doc_embedder.documents", "doc_writer.documents")
        indexing_pipe.run({"doc_embedder": {"documents": docs}})
    else:
        document_store.write_documents(docs)

    template = """
    You are a top-tier mathematician who is exceptionally skilled at predicting missing values of a symmetric sparse matrix.
    Note that the values of a symmetric sparse matrix are all decimal numbers ranging between 0 and 1.
    The current task is to predict the decimal value at a specific position in the matrix (identified by its row and column indices) based on the provided examples.
    Known Value Examples:
    {% for doc in documents %}
        {{ doc.content }}
    {% endfor %}

    {{my_query}}
    
    Answer:
    """
    query_template_base = """
    Based on above known values, 
    {0}
    Please reason step by step, and predict its value.
    Please put your predicted value between two $ symbols. 
    """
    pattern = r'[\s\S]*\$[\s\S]*(\d\.\d+)[\s\S]*\$[\s\S]*'
    for query, target in zip(queries, targets):
        new_query = query_template_base.format(query)
        success_flag, result = predict(document_store, new_query, template, tokenizer, model, pattern, embedding_flag, multi_flag, n_retrieves)
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
            success_results)) if len(success_results) != len(success_targets) else print("metric computation")
        success_results = np.array(success_results)
        success_targets = np.array(success_targets)
        rmse = RMSE(success_results, success_targets)
        print("rmse:", rmse)
        mae = MAE(success_results, success_targets)
        print("mae:", mae)



