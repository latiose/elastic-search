
import bm25s # To create indexes and search documents using BM25
import Stemmer
import json  # To load the JSON-formatted corpus
import gdown
import zipfile
import os
import csv
import chromadb
import json
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

link = "https://drive.google.com/uc?id=1R2_dRaKWNzI72vdQayPgoIJ9q6Hv8Uzg&export=download" 
archivo = "trec-covid-RI.zip"
if not os.path.exists(archivo):
    gdown.download(link,archivo)
with zipfile.ZipFile(archivo, 'r') as zip_ref:
    zip_ref.extractall("trec-covid-RI")


def get_batches(lista, chunk_size=100):
    return [lista[i:i + chunk_size] for i in range(0, len(lista), chunk_size)]

corpus_content = []
with open("trec-covid-RI/corpus.jsonl","r",encoding="utf-8") as f:
    for line in f:
        corpus_content.append(json.loads(line))
        
def descargar():
    link = "https://drive.google.com/uc?id=1Me1JIbrMOJtHLj7e0u_yjfaQnZM0BwvN&export=download"
    archivo = "chromadb-storage.zip"
    gdown.download(link,archivo)
    with zipfile.ZipFile(archivo, 'r') as zip_ref:
        zip_ref.extractall("chromadb-storage")


def crear():
    chromadb_documents = []
    chromadb_doc_ids = []
    
    for document in corpus_content:
    #print(document)
        doc_id = str(document["_id"])
        title = document["title"].lower()
        content = document["text"].lower()

        chromadb_doc_ids.append(doc_id)
        chromadb_documents.append(f"{title} {content}")

    chromadb_embeddings = model.encode(chromadb_documents, batch_size=100, show_progress_bar=True)#, device='cuda')
   
   
    document_batches = get_batches(chromadb_documents)
    ids_batches = get_batches(chromadb_doc_ids)
    embedding_batches = get_batches(chromadb_embeddings)
    for i in range(len(document_batches)):
       documents = document_batches[i]
       doc_ids = ids_batches[i]
       embeddings = embedding_batches[i]
       collection.add(documents=documents,ids=doc_ids,embeddings=embeddings)

def submit_queries_and_get_run(queries, collection, max_results, nombre):
    run = {}
    for query in queries:
        query_id = query["_id"]
        query_string = query["metadata"]["query"].lower()
        
        results = collection.query(
            query_texts=[query_string],
            n_results=max_results
        )
        run[query_id] = results['ids'][0]
        
     
    compute_precision_recall_f1(run, relevance_judgements_reformat, nombre)


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

    with open(f"{nombre} medidas.json", "w", encoding="utf-8") as f:
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
all_queries_spanish = list()
with open("trec-covid-RI/queries_spanish.jsonl", "r", encoding="utf-8") as f:
     for line in f:
        all_queries_spanish.append(json.loads(line))
all_queries_paraphrased = list()
with open("trec-covid-RI/queries_paraphrased.jsonl", "r", encoding="utf-8") as f:
     for line in f:
        all_queries_paraphrased.append(json.loads(line))



# Initialize the sentence transformer model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Create a persistent ChromaDB client
client = chromadb.PersistentClient(path="./chromadb-storage/")

# We create the collection, please note how we are providing the embedding
# pre-trained model (this is a multilingual model) and we specify the
# distance metric to find the nearest neighbors
existing_collections = client.list_collections()
collection_name = "COVID_collection"
existe = False
if not collection_name in [col.name for col in existing_collections]:
    collection = client.create_collection(
    name=collection_name,
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    metadata={"hnsw:space": "cosine"} # https://docs.trychroma.com/guides#changing-the-distance-function
)
else:
    collection = client.get_collection(collection_name,embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
    existing_ids = collection.get()["ids"]
    existe = True
    
    submit_queries_and_get_run(all_queries, collection, 100,"")

    submit_queries_and_get_run(all_queries_paraphrased, collection, 100,"paraphrased")

    submit_queries_and_get_run(all_queries_spanish, collection, 100,"spanish")

    #print(f"The collection {collection_name} contains {len(existing_ids)} documents")

if(not existe):
   descargar()
   print("Vuelva a ejecutar el script para conseguir los índices F1")



