
import bm25s # To create indexes and search documents using BM25
import Stemmer
import json  # To load the JSON-formatted corpus
import gdown
import zipfile
import os
import math
from collections import Counter
from heapq import heappush, heappop
import csv
import copy
from typing import TypeVar, List, Generic
from dataclasses import dataclass
from collections import Counter
from heapq import heappush, heappop
import math

class LogLikelihood:
    @staticmethod
    def entropy(*elements):
      
        if any(e < 0 for e in elements):
            raise ValueError("All elements must be non-negative")

        total_sum = sum(elements)
        result = sum(LogLikelihood.x_log_x(e) for e in elements)
        return LogLikelihood.x_log_x(total_sum) - result

    @staticmethod
    def x_log_x(x):
        """
        Helper function to calculate x * log(x), handling x = 0.
        """
        return 0.0 if x == 0 else x * math.log(x)

    @staticmethod
    def log_likelihood_ratio(k11, k12, k21, k22):
        """
        Calculates the log-likelihood ratio for two events.
        """
        assert all(k >= 0 for k in (k11, k12, k21, k22)), "Counts must be non-negative"
        
        row_entropy = LogLikelihood.entropy(k11 + k12, k21 + k22)
        column_entropy = LogLikelihood.entropy(k11 + k21, k12 + k22)
        matrix_entropy = LogLikelihood.entropy(k11, k12, k21, k22)
        
        if row_entropy + column_entropy < matrix_entropy:
            # Handle potential floating-point rounding errors
            return 0.0
        
        return 2.0 * (row_entropy + column_entropy - matrix_entropy)

    @staticmethod
    def root_log_likelihood_ratio(k11, k12, k21, k22):
        """
        Calculates the root log-likelihood ratio for two events.
        """
        llr = LogLikelihood.log_likelihood_ratio(k11, k12, k21, k22)
        sqrt_llr = math.sqrt(llr)
        if(k21 + k22>0):
            if (k11 / (k11 + k12)) < (k21 / (k21 + k22)):
             sqrt_llr = -sqrt_llr
        
        return sqrt_llr

    @staticmethod
    def compare_frequencies(a, b, max_return, threshold):
        """
        Compares two sets of counts to find items over-represented in the first set.
        """
        total_a = sum(a.values())
        total_b = sum(b.values())
        
        best = []  # Min-heap for top `max_return` scored items

        # Compare elements in `a`
        for item in a:
            LogLikelihood._compare_and_add(a, b, max_return, threshold, total_a, total_b, best, item)

        # Compare elements only in `b` if threshold < 0
        if threshold < 0:
            for item in b:
                if item not in a:
                    LogLikelihood._compare_and_add(a, b, max_return, threshold, total_a, total_b, best, item)

        # Sort results by score in descending order
        return sorted(best, key=lambda x: -x[1])

    @staticmethod
    def _compare_and_add(a, b, max_return, threshold, total_a, total_b, best, item):
        """
        Helper method to compute the score for an item and add it to the heap.
        """
        k_a = a.get(item, 0)
        k_b = b.get(item, 0)
        
        score = LogLikelihood.root_log_likelihood_ratio(k_a, total_a - k_a, k_b, total_b - k_b)
        if score >= threshold:
            heappush(best, (item, score))
            if len(best) > max_return:
                heappop(best)


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

def compute_term_frequencies_from_corpus_tokenized(corpus_tokenized):

    tmp = dict()

    for document in corpus_tokenized[0]:
        freqs = dict(Counter(document))
        for token, freq in freqs.items():
            try:
                tmp[token] += freq
            except:
                tmp[token] = freq


    total_freqs = dict()

    for key, freq in dict(tmp).items():
        term = tmp[key]
        total_freqs[term] = freq

    return total_freqs

def calculate_term_frequencies_from_run(run):
    import re
    tmp = dict()

    for cuerpo in run["cuerpos"]:  
        tokens = re.findall(r'\b\w+\b', cuerpo.lower())  
        freqs = dict(Counter(tokens)) 
        
        for token, freq in freqs.items():
            try:
                tmp[token] += freq
            except KeyError:
                tmp[token] = freq

    return tmp 

def indexar():
        corpus_tokenized = bm25s.tokenize(corpus_plaintext, stopwords="en",stemmer=stemmer, show_progress=True)
        retriever = bm25s.BM25(corpus=corpus_verbatim, method="bm25+", idf_method="bm25+")
        retriever.index(corpus_tokenized, show_progress=True)
        term_frequencies_totales = compute_term_frequencies_from_corpus_tokenized(corpus_tokenized)
        submit_queries_and_get_run(all_queries,stemmer,retriever,n,term_frequencies_totales,"expansion")

def submit_queries_and_get_run(queries, stemmer, retriever, max_results,term_frequencies_totales,carpeta):
        run2 = {}
        for query in queries:
        
            run = {  
                "ids": [],  
                "cuerpos": []  
            }

            query_id = query["_id"]

            query_string = query["metadata"]["query"].lower()
         
            query_tokenized = bm25s.tokenize(query_string, stopwords="en",stemmer=stemmer, show_progress=False)
            results = retriever.retrieve(query_tokenized,corpus=retriever.corpus, k=max_results, return_as="tuple", show_progress=False)
            returned_documents = results.documents[0]
            for doc in returned_documents:
                doc_id = str(doc["id"])
                cuerpo = doc["text"]
                
                run["ids"].append(doc_id)
                run["cuerpos"].append(cuerpo)
            
            term_frequencies_run = calculate_term_frequencies_from_run(run)
            #print(term_frequencies_run)
            term_frequencies_resto = copy.deepcopy(term_frequencies_totales)
            for term, freq in term_frequencies_run.items():
                if term in term_frequencies_resto:
                    term_frequencies_resto[term] = term_frequencies_resto[term] - freq
          
            results = LogLikelihood.compare_frequencies(term_frequencies_run, term_frequencies_resto, sum([1 for item in term_frequencies_resto]), 0)
            #print(term_frequencies_resto)
            i=0
            resultados = []
            #print(results)
            for elemento in results:
                if i == m:
                    break
                if(elemento[0] not in query_string): 
                    resultados.append(elemento[0])  
                    i += 1
     
            consulta_modificada = query_string + " " + " ".join(resultados)
            #print(consulta_modificada)
            query_tokenized = bm25s.tokenize(consulta_modificada.lower(),stopwords="en", stemmer=stemmer, show_progress=False)
            results = retriever.retrieve(query_tokenized, corpus=retriever.corpus, k=100, return_as="tuple", show_progress=False)
                        #print(results)
            returned_documents = results.documents[0]
                        
            returned_ids = []
            for i in range(len(returned_documents)):
                    returned_ids.append(str(returned_documents[i]["id"]))
            run2[query_id] = returned_ids
        nombre = f"m={m}n={max_results}"

        compute_precision_recall_f1(run2,relevance_judgements_reformat,nombre,carpeta)


def compute_precision_recall_f1(run, relevance_judgements, nombre,carpeta):
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
    os.makedirs(carpeta, exist_ok=True)
    with open(f"{carpeta}/{nombre}medidas.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

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


all_queries = list()
with open("trec-covid-RI/queries.jsonl", "r", encoding="utf-8") as f:
     for line in f:
        all_queries.append(json.loads(line))


n= int(input("¿Cuántos documentos quieres? "))
m= int(input("¿Cuántos términos quieres añadir? "))
indexar()

