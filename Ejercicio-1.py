
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

corpus_tokenized = bm25s.tokenize(corpus_plaintext, stopwords="en", stemmer=stemmer, show_progress=True)

bm25_flavor = "lucene"
idf_flavor = "lucene"

retriever = bm25s.BM25(corpus=corpus_verbatim, method=bm25_flavor, idf_method=idf_flavor)
retriever.index(corpus_tokenized, show_progress=True)

retriever.save("trec", corpus=corpus_verbatim)