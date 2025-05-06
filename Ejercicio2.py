import csv
import bm25s # To create indexes and search documents using BM25
import Stemmer
import json  # To load the JSON-formatted corpus
import gdown
import zipfile
import os

link = "https://drive.google.com/uc?id=1R2_dRaKWNzI72vdQayPgoIJ9q6Hv8Uzg&export=download" 
archivo = "trec-covid-RI.zip"
if not os.path.exists(archivo):
    gdown.download(link,archivo)
with zipfile.ZipFile(archivo, 'r') as zip_ref:
    zip_ref.extractall("trec-covid-RI")

corpus_content = []
with open("trec-covid-RI/corpus.jsonl","r",encoding="utf-8") as f:
    for line in f:
        corpus_content.append(json.loads(line))

corpus_verbatim = list()
corpus_plaintext = list()
for entry in corpus_content:
        document = {"id": entry["_id"], "title": entry["title"].lower(), "text": entry["text"].lower(),"url": entry["metadata"]["url"],"pubmed_id": entry["metadata"]["pubmed_id"]}
        corpus_verbatim.append(document)
        corpus_plaintext.append(f"{entry['title'].lower()} {entry['text'].lower()}")

stemmer = Stemmer.Stemmer("english")

all_queries = list()
with open("trec-covid-RI/queries.jsonl", "r", encoding="utf-8") as f:
     for line in f:
        all_queries .append(json.loads(line))
#print(all_queries)

bm25_flavours = ["lucene","robertson","atire","bm25l","bm25+"]

qrels_file = "trec-covid-RI/qrels.tsv"  
relevance_judgements_reformat = {}

with open(qrels_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    next(reader)  
    for row in reader:
        query_id, corpus_id, score = row
        if query_id not in relevance_judgements_reformat:
            relevance_judgements_reformat[query_id] = []
        relevance_judgements_reformat[query_id].append(corpus_id)
#print(relevance_judgements_reformat)

possible_names = [
    f"{flavor}-{sw}-{st}"
    for flavor in bm25_flavours
    for sw in ["stopwords", "NON-stopwords"]
    for st in ["stemming", "NON-stemming"]
]


def descargar():
    link = "https://drive.google.com/uc?id=1AVJQxu8m980wqetdl_BxWLas6QNvflrE&export=download"
    archivo = "colecciones.zip"
    gdown.download(link,archivo)
    with zipfile.ZipFile(archivo, 'r') as zip_ref:
        zip_ref.extractall("colecciones")

def indexar():
   for bm25_flavor in bm25_flavours:
            for stopwords in [True, False]: 
                for stem in [True, False]:  
                    if stopwords:
                        if stem:
                            corpus_tokenized = bm25s.tokenize(corpus_plaintext, stopwords="en", stemmer=stemmer, show_progress=True)
                        else:
                            corpus_tokenized = bm25s.tokenize(corpus_plaintext, stopwords="en", show_progress=True)
                    else:
                        if stem:
                            corpus_tokenized = bm25s.tokenize(corpus_plaintext, stemmer=stemmer, show_progress=True)
                        else:
                            corpus_tokenized = bm25s.tokenize(corpus_plaintext,  show_progress=True)
                    stopwords_str = "stopwords" if stopwords else "NON-stopwords"
                    stemming_str = "stemming" if stem else "NON-stemming"
                    file_name = f"{bm25_flavor}-{stopwords_str}-{stemming_str}"
                    retriever = bm25s.BM25(corpus=corpus_verbatim, method=bm25_flavor, idf_method=bm25_flavor)
                    retriever.index(corpus_tokenized, show_progress=True)
                    retriever.save(file_name, corpus=corpus_verbatim)
                    submit_queries_and_get_run(all_queries,stemmer,retriever,stopwords,stem,file_name)

def submit_queries_and_get_run(queries, stemmer, retriever,stopwords,stem,nombre, max_results=100):
    run = {}
    for query in queries:
        query_id = query["_id"]

        query_string = query["metadata"]["query"].lower()

        if stopwords:
                    if stem:
                            query_tokenized = bm25s.tokenize(query_string, stopwords="en", stemmer=stemmer, show_progress=False)
                    else:
                            query_tokenized = bm25s.tokenize(query_string, stopwords="en", show_progress=False)
        else:
                        if stem:
                            query_tokenized = bm25s.tokenize(query_string, stemmer=stemmer, show_progress=False)
                        else:
                            query_tokenized = bm25s.tokenize(query_string, show_progress=False)
                      
        results = retriever.retrieve(query_tokenized, corpus=retriever.corpus, k=max_results, return_as="tuple", show_progress=False)
        returned_documents = results.documents[0]
        #print(returned_documents)
        returned_ids = []
        for i in range(len(returned_documents)):
            returned_ids.append(str(returned_documents[i]["id"]))
        run[query_id] = returned_ids
       # print(returned_ids)
    compute_precision_recall_f1(run,relevance_judgements_reformat,nombre)

def submit_queries_and_get_run2(queries, stemmer, retriever,stopwords,stem,nombre, max_results=100):
    run = {}
    for query in queries:
        query_id = query["_id"]

        query_string = query["metadata"]["query"].lower()

        if stopwords:
                    if stem:
                            query_tokenized = bm25s.tokenize(query_string, stopwords="en", stemmer=stemmer, show_progress=False)
                    else:
                            query_tokenized = bm25s.tokenize(query_string, stopwords="en", show_progress=False)
        else:
                        if stem:
                            query_tokenized = bm25s.tokenize(query_string, stemmer=stemmer, show_progress=False)
                        else:
                            query_tokenized = bm25s.tokenize(query_string, show_progress=False)
                      
        results = retriever.retrieve(query_tokenized, corpus=corpus_verbatim, k=max_results, return_as="tuple", show_progress=False)
        returned_documents = results.documents[0]
        #print(returned_documents)
        returned_ids = []
        for i in range(len(returned_documents)):
            returned_ids.append(str(returned_documents[i]["id"]))
        run[query_id] = returned_ids
       # print(returned_ids)
    compute_precision_recall_f1(run,relevance_judgements_reformat,nombre)



def compute_precision_recall_f1(run, relevance_judgements, nombre):
    results = {}

    # Initialize lists to hold precision, recall, and f1 scores for each query
    precision_values = []
    recall_values = []
    f1_values = []

    # Initialize global counts for micro-averaging
    global_retrieved = 0
    global_relevant = 0
    global_retrieved_and_relevant = 0

    # Compute precision, recall, and F1 score for each query
    for query_id in run.keys():
        retrieved_results = run[query_id]
        relevant_results = relevance_judgements[query_id]
        relevant_and_retrieved = set(retrieved_results) & set(relevant_results)

        # Update global counts
        global_retrieved += len(retrieved_results)
        global_relevant += len(relevant_results)
        global_retrieved_and_relevant += len(relevant_and_retrieved)

        # Compute precision and recall
        precision = len(relevant_and_retrieved) / len(retrieved_results) if len(retrieved_results) > 0 else 0
        recall = len(relevant_and_retrieved) / len(relevant_results) if len(relevant_results) > 0 else 0

        # Compute F1 score if both precision and recall are non-zero
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Append precision and recall for the current query
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

        results[query_id] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        }

    # Compute macro-averages
    macro_average_precision = sum(precision_values) / len(precision_values) if precision_values else 0
    macro_average_recall = sum(recall_values) / len(recall_values) if recall_values else 0
    macro_average_f1 = sum(f1_values) / len(f1_values) if f1_values else 0

    # Compute micro-averages
    micro_average_precision = global_retrieved_and_relevant / global_retrieved if global_retrieved > 0 else 0
    micro_average_recall = global_retrieved_and_relevant / global_relevant if global_relevant > 0 else 0
    micro_average_f1 = (2 * (micro_average_precision * micro_average_recall) /
                        (micro_average_precision + micro_average_recall)) if (micro_average_precision + micro_average_recall) > 0 else 0

    results["macro_averages"] = {
        "precision": round(macro_average_precision, 3),
        "recall": round(macro_average_recall, 3),
        "f1": round(macro_average_f1, 3),
    }

    results["micro_averages"] = {
        "precision": round(micro_average_precision, 3),
        "recall": round(micro_average_recall, 3),
        "f1": round(micro_average_f1, 3),
    }

    with open(f"colecciones/{nombre}/medidas.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)



#indexar()


def descargarTodo():
    for name in possible_names:
        directory_path = os.path.join("colecciones", name)  
        if not os.path.isdir(directory_path):
            descargar()
    #print(name)
    retriever = bm25s.BM25.load(directory_path)
    stopwords = "NON-stopwords" not in name
    stem = "NON-stemming" not in name
    submit_queries_and_get_run2(all_queries,stemmer,retriever,stopwords,stem,name)

descargarTodo()