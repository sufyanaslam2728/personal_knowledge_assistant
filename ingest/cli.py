# ingest/cli.py
import argparse
import os
from pathlib import Path
from typing import Iterable

from ingest.loaders import load_documents
from ingest.chunker import chunk_documents
from ingest.embedder import Embedder
from ingest.indexer import FaissStore

def iter_supported_files(path: str) -> Iterable[str]:
    p = Path(path)
    if p.is_dir():
        exts = ("*.pdf", "*.docx", "*.pptx", "*.ppt", "*.txt", "*.md")
        for ext in exts:
            for f in p.rglob(ext):
                yield str(f)
    elif p.exists():
        yield str(p)
    else:
        raise FileNotFoundError(f"Path not found: {path}")

def main():
    parser = argparse.ArgumentParser(
        description="Ingest files → chunk → embed → index (FAISS)."
    )
    parser.add_argument("--path", required=True, help="File or folder to ingest.")
    parser.add_argument(
        "--index",
        default=os.getenv("INDEX_PATH", "/app/data/indices/faiss.index"),
        help="FAISS index file path inside container.",
    )
    parser.add_argument(
        "--store",
        default=os.getenv("STORE_PATH", "/app/data/indices/store.pkl"),
        help="Pickle store for texts/metadata.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        help="SentenceTransformer embedding model.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("CHUNK_SIZE", 800)),
        help="Chunk size (chars).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=int(os.getenv("CHUNK_OVERLAP", 100)),
        help="Chunk overlap (chars).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Do not load existing index; start new.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional: test a query against the built/loaded index.",
    )
    parser.add_argument("--k", type=int, default=5, help="Top-k results for --query.")
    args = parser.parse_args()

    embedder = Embedder(model_name=args.model)
    store = FaissStore(index_path=args.index, store_path=args.store)

    if not args.fresh and os.path.exists(args.index) and os.path.exists(args.store):
        store.load()

    total_chunks = 0
    texts, metas = [], []

    for file in iter_supported_files(args.path):
        docs = load_documents(file)
        chunks = chunk_documents(docs, args.chunk_size, args.chunk_overlap)
        texts.extend([c.page_content for c in chunks])
        metas.extend([c.metadata for c in chunks])
        total_chunks += len(chunks)

    if total_chunks == 0:
        print("No supported documents found.")
        return

    vectors = embedder.embed_texts(texts)
    store.add(vectors, texts, metas)
    store.save()

    print(f"✅ Indexed {total_chunks} chunks into:")
    print(f"    FAISS: {args.index}")
    print(f"    Store: {args.store}")

    if args.query:
        qv = embedder.embed_query(args.query)
        results = store.search(qv, k=args.k)
        print("\n🔎 Query results:")
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            src = meta.get("source", "?")
            page = meta.get("page") or meta.get("slide")
            loc = f" (page {page})" if page else ""
            print(f"{i}. score={r['score']:.4f} | {src}{loc}\n   {r['text'][:300]}...\n")

if __name__ == "__main__":
    main()
