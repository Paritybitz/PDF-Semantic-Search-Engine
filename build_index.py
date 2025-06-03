import os
import json
from pathlib import Path

import faiss
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


def collect_chunks(users_dir: Path):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    metadata = []
    texts = []

    for user_dir in users_dir.iterdir():
        if not user_dir.is_dir():
            continue
        for pdf_path in user_dir.glob("*.pdf"):
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            docs = text_splitter.split_documents(pages)
            for idx, doc in enumerate(docs):
                text = doc.page_content
                texts.append(text)
                metadata.append({
                    "user": user_dir.name,
                    "file": pdf_path.name,
                    "chunk_id": idx,
                    "chunk": text,
                })
    return texts, metadata


def build_index(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = embeddings.astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


def main():
    users_dir = Path("users")
    index_path = Path("index.faiss")
    meta_path = Path("index_meta.json")

    texts, metadata = collect_chunks(users_dir)
    index = build_index(texts)

    faiss.write_index(index, str(index_path))
    with meta_path.open("w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    main()
