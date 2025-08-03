# coding=utf-8

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
# from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchEmbeddingRetriever
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever, InMemoryBM25Retriever
from haystack.components.builders import PromptBuilder

def reciprocal_rank_fusion(docs_list, k=60):
    rrf_scores_dict = {doc.id: {'rrf_score': 0} for docs in docs_list for doc in docs}
    for docs in docs_list:
        for rank, doc in enumerate(docs):
            rrf_scores_dict[doc.id]['rrf_score'] += 1 / (rank + 1 + k)
            if 'document' not in rrf_scores_dict[doc.id]:
                rrf_scores_dict[doc.id]['document'] = doc
    sorted_list = sorted(rrf_scores_dict.items(), key=lambda x: x[1]['rrf_score'], reverse=True)
    return sorted_list

def BM25_embedding_pipeline(document_store, query, template, n_retrieves=3):
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    # embedding_retriever = ElasticsearchEmbeddingRetriever(document_store=document_store, top_k=n_retrieves)
    embedding_retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k=n_retrieves)
    BM25_retriever = InMemoryBM25Retriever(document_store=document_store, top_k=n_retrieves)
    prompt_builder = PromptBuilder(template=template)
    text_embedder.warm_up()
    query_embedding_result = text_embedder.run(text=query)
    embedding_retriever_result = embedding_retriever.run(query_embedding=query_embedding_result['embedding'])
    BM25_retriever_result = BM25_retriever.run(query=query)
    retriever_docs_list = []
    retriever_docs_list.append(embedding_retriever_result['documents'])
    retriever_docs_list.append(BM25_retriever_result['documents'])
    sorted_list = reciprocal_rank_fusion(retriever_docs_list)
    sorted_docs = [sorted_list[i][1]['document'] for i in range(n_retrieves)]
    prompt_result = prompt_builder.run(documents=sorted_docs, my_query=query)
    return prompt_result['prompt']

def embedding_pipeline(document_store, query, template, n_retrieves=3):
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    # embedding_retriever = ElasticsearchEmbeddingRetriever(document_store=document_store, top_k=n_retrieves)
    retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k=n_retrieves)
    prompt_builder = PromptBuilder(template=template)
    pipe = Pipeline()
    pipe.add_component(name="query_embedder", instance=text_embedder)
    pipe.add_component(name="retriever", instance=retriever)
    pipe.add_component(name="prompt_builder", instance=prompt_builder)
    pipe.connect("query_embedder.embedding", "retriever.query_embedding")
    pipe.connect("retriever.documents", "prompt_builder.documents")
    prompt_result = pipe.run({"query_embedder": {"text": query},
                              "prompt_builder": {"my_query": query}})
    return prompt_result['prompt_builder']['prompt']

def BM25_pipeline(document_store, query, template, n_retrieves=3):
    retriever = InMemoryBM25Retriever(document_store=document_store, top_k=n_retrieves)
    prompt_builder = PromptBuilder(template=template)
    pipe = Pipeline()
    pipe.add_component(name="retriever", instance=retriever)
    pipe.add_component(name="prompt_builder", instance=prompt_builder)
    pipe.connect("retriever.documents", "prompt_builder.documents")
    prompt_result = pipe.run({"retriever": {"query": query},
                              "prompt_builder": {"my_query": query}})
    return prompt_result['prompt_builder']['prompt']