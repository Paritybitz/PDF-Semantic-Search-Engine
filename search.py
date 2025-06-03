import json
import sys
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
import numpy as np


INDEX_FILE = Path("index.faiss")
META_FILE = Path("index_meta.json")


def load_resources():
    index = faiss.read_index(str(INDEX_FILE))
    with META_FILE.open() as f:
        metadata = json.load(f)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, metadata, model


def search(query: str, k: int = 5):
    index, metadata, model = load_resources()
    emb = model.encode([query]).astype("float32")
    distances, ids = index.search(emb, k)

    for dist, idx in zip(distances[0], ids[0]):
        info = metadata[idx]
        print(f"File: {info['user']}/{info['file']} - Chunk {info['chunk_id']} (score: {dist:.4f})")
        print(info['chunk'])
        print("-" * 40)


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter query: ")
    search(query)


if __name__ == "__main__":
    main()
