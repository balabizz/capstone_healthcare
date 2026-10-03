"""Ingest general-reference PDFs into the shared medical RAG index."""
import argparse
import json
from pathlib import Path
from src.config import FAISS_INDEX_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from src.data_processing.ingestion import DocumentIngestion, MAX_UPLOAD_BYTES
from src.vector_store.faiss_store import FAISSStore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='*', help='PDF files or directories (top-level PDFs)')
    parser.add_argument('--index-path', default=FAISS_INDEX_PATH)
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE)
    parser.add_argument('--chunk-overlap', type=int, default=CHUNK_OVERLAP)
    parser.add_argument('--list', action='store_true', help='List indexed documents without calling OpenAI')
    parser.add_argument('--search', help='Preview semantic search after ingestion')
    args = parser.parse_args(argv)
    if not args.paths and not args.list and args.search is None:
        parser.error('Supply PDF files/directories, --list, or --search.')
    try:
        store = FAISSStore(args.index_path)
        ingestion = DocumentIngestion(store, args.chunk_size, args.chunk_overlap)
    except ValueError as error:
        parser.error(str(error))
    failed = False
    for value in args.paths:
        path = Path(value)
        files = sorted(p for p in path.iterdir() if p.suffix.lower() == '.pdf') if path.is_dir() else [path]
        if not files:
            print(json.dumps({'source': str(path), 'status': 'failed', 'error': 'No PDFs found in directory.'}))
            failed = True
        for pdf in files:
            try:
                with pdf.open('rb') as stream:
                    data = stream.read(MAX_UPLOAD_BYTES + 1)
                print(json.dumps(ingestion.ingest(pdf.name, data)))
            except (ValueError, OSError) as error:
                print(json.dumps({'source': pdf.name, 'status': 'failed', 'error': str(error)}))
                failed = True
            except Exception:
                print(json.dumps({'source': pdf.name, 'status': 'failed', 'error': 'Indexing failed or index is busy; retry.'}))
                failed = True
    try:
        if args.list:
            print(json.dumps(store.list_documents(), indent=2))
        if args.search is not None:
            print(json.dumps([{'text': doc.page_content, 'metadata': doc.metadata, 'distance': float(score)}
                              for doc, score in store.search(args.search)], indent=2))
    except Exception:
        print(json.dumps({'status': 'failed', 'error': 'Could not read/search index. Check index path and embedding configuration.'}))
        failed = True
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
