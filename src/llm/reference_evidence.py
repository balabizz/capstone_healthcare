"""Deterministic labels for retrieved documents; no model-invented source metadata."""


def reference_evidence(documents):
    evidence = []
    for document in documents:
        text = getattr(document, 'page_content', None)
        if not isinstance(text, str) or not text.strip():
            continue
        metadata = getattr(document, 'metadata', {}) or {}
        source = str(metadata.get('source') or 'Unknown source')
        page = metadata.get('page')
        evidence.append({'id': f'ref:{len(evidence)+1}', 'source': source,
                         'page': page, 'chunk_index': metadata.get('chunk_index'),
                         'document_id': metadata.get('document_id'), 'excerpt': text})
    return evidence


def reference_notice(evidence):
    """These are retrieval provenance, not validated claim-level citations."""
    def plain(value):
        return ' '.join(str(value).split()).translate(str.maketrans({c: ' ' for c in '`[]<>*\\'}))[:300]
    return 'Retrieved reference documents:\n' + '\n'.join(
        f"- [{r['id']}] {plain(r['source'])}" +
        (f" — page {plain(r['page'])}" if r['page'] is not None else ' — page not supplied')
        for r in evidence)
