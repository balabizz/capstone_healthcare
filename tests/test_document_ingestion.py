from io import BytesIO
from unittest.mock import Mock
import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject, DictionaryObject
from langchain_core.embeddings import Embeddings
from src.data_processing.ingestion import DocumentIngestion, MAX_UPLOAD_BYTES
from src.vector_store.faiss_store import FAISSStore


def pdf_bytes(text='Kidney disease treatment reference.', encrypted=False, blank=False):
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    if not blank:
        font = DictionaryObject({NameObject('/Type'):NameObject('/Font'),NameObject('/Subtype'):NameObject('/Type1'),NameObject('/BaseFont'):NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):writer._add_object(font)})})
        stream = DecodedStreamObject()
        stream.set_data(f'BT /F1 12 Tf 10 200 Td ({text}) Tj ET'.encode())
        page[NameObject('/Contents')] = writer._add_object(stream)
    if encrypted:
        writer.encrypt('password')
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


class Vectors(Embeddings):
    model = 'test-model'
    def __init__(self):
        self.calls = 0
        self.fail = False
    def embed_documents(self,texts):
        self.calls += 1
        if self.fail:
            raise ValueError('Embedding unavailable')
        return [[float('kidney' in t.lower()),float('asthma' in t.lower()),1.] for t in texts]
    def embed_query(self,text):
        return self.embed_documents([text])[0]


@pytest.fixture
def pipeline(tmp_path):
    return DocumentIngestion(FAISSStore(tmp_path/'index',Vectors()),chunk_size=100,chunk_overlap=10)


def test_real_pdf_to_persisted_faiss_and_rag_retriever(pipeline):
    result = pipeline.ingest('../guideline.pdf',pdf_bytes())
    assert result['status'] == 'indexed' and result['pages'] == 1
    reopened = FAISSStore(pipeline.store.index_path,Vectors())
    matches = reopened.search('kidney')
    assert matches[0][0].metadata['source'] == 'guideline.pdf'
    assert matches[0][0].metadata['page'] == 1
    assert 'Kidney' in reopened.load_store().as_retriever().invoke('kidney')[0].page_content
    assert reopened.list_documents()[0]['sha256'] == result['sha256']


def test_duplicate_skips_embeddings_and_append_preserves_sources(pipeline):
    data = pdf_bytes()
    pipeline.ingest('one.pdf',data)
    calls = pipeline.store.embeddings.calls
    assert pipeline.ingest('renamed.pdf',data)['status'] == 'duplicate'
    assert pipeline.store.embeddings.calls == calls
    pipeline.ingest('two.pdf',pdf_bytes('Asthma information.'))
    assert len(pipeline.store.list_documents()) == 2
    assert pipeline.store.search('asthma',k=1)[0][0].metadata['source'] == 'two.pdf'


@pytest.mark.parametrize('name,data',[
    ('bad.txt',b'x'),('bad.pdf',b'not a pdf'),('empty.pdf',b''),
    ('broken.pdf',b'%PDF-broken'),('scan.pdf',pdf_bytes(blank=True)),
    ('secret.pdf',pdf_bytes(encrypted=True))])
def test_invalid_upload_never_embeds(pipeline,name,data):
    with pytest.raises(ValueError):
        pipeline.ingest(name,data)
    assert pipeline.store.embeddings.calls == 0
    assert pipeline.store.list_documents() == []


def test_embedding_failure_preserves_existing_index(pipeline):
    pipeline.ingest('one.pdf',pdf_bytes())
    pipeline.store.embeddings.fail = True
    with pytest.raises(ValueError):
        pipeline.ingest('two.pdf',pdf_bytes('Other new content'))
    assert len(pipeline.store.list_documents()) == 1
    pipeline.store.embeddings.fail = False
    assert len(pipeline.store.search('kidney')) == 1


def test_commit_failure_rolls_back_index_and_registry(pipeline,monkeypatch):
    pipeline.ingest('one.pdf',pdf_bytes())
    save = pipeline.store._save
    def failed(c,store,manifest):
        save(c,store,manifest)
        raise ValueError('Simulated commit failure')
    monkeypatch.setattr(pipeline.store,'_save',failed)
    with pytest.raises(ValueError):
        pipeline.ingest('two.pdf',pdf_bytes('Other new content'))
    assert len(pipeline.store.list_documents()) == 1
    assert len(pipeline.store.search('kidney')) == 1


def test_limits_and_empty_index(pipeline):
    with pytest.raises(ValueError,match='20 MB'):
        pipeline.ingest('large.pdf',b'x'*(MAX_UPLOAD_BYTES+1))
    with pytest.raises(ValueError,match='No reference index'):
        pipeline.store.load_store()
    with pytest.raises(ValueError):
        DocumentIngestion(pipeline.store,chunk_size=100,chunk_overlap=100)


def test_cli_ingests_lists_and_reports_failure(pipeline,tmp_path,monkeypatch,capsys):
    import scripts.ingest_documents as cli
    monkeypatch.setattr(cli,'FAISSStore',lambda path: pipeline.store)
    path = tmp_path/'guide.pdf'
    path.write_bytes(pdf_bytes())
    assert cli.main([str(path),'--list','--search','kidney']) == 0
    assert 'indexed' in capsys.readouterr().out
    assert cli.main([str(tmp_path/'missing.pdf')]) == 1


def test_incompatible_model_does_not_modify_index(pipeline):
    pipeline.ingest('one.pdf',pdf_bytes())
    pipeline.store.embeddings.model = 'different'
    with pytest.raises(ValueError,match='model differs'):
        pipeline.ingest('two.pdf',pdf_bytes('New material'))
    assert len(pipeline.store.list_documents()) == 1


def test_separate_writer_instances_preserve_both_documents(pipeline):
    second = DocumentIngestion(FAISSStore(pipeline.store.index_path,Vectors()))
    pipeline.ingest('one.pdf',pdf_bytes())
    second.ingest('two.pdf',pdf_bytes('Asthma information.'))
    assert len(pipeline.store.list_documents()) == 2
    assert len(pipeline.store.search('asthma')) == 2


def test_staff_upload_screen_and_patient_access(pipeline,monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import importlib.util
    import sys
    from pathlib import Path
    messages = []
    ui = SimpleNamespace(
        expander=lambda *a,**k:nullcontext(), spinner=lambda *a,**k:nullcontext(),
        caption=lambda *a,**k:None, dataframe=lambda *a,**k:None,
        success=lambda msg:messages.append(msg), error=lambda msg:messages.append(msg),
        warning=lambda msg:messages.append(msg), info=lambda msg:messages.append(msg),
        file_uploader=lambda *a,**k:[SimpleNamespace(name='guide.pdf',getvalue=pdf_bytes)],
        button=lambda label:label == 'Index uploaded PDFs', text_input=lambda label:'',
    )
    monkeypatch.setitem(sys.modules,'streamlit',ui)
    spec = importlib.util.spec_from_file_location('ingestion_view_test',Path('app/document_ingestion_view.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.render_document_ingestion('patient',pipeline)
    assert pipeline.store.list_documents() == []
    module.render_document_ingestion('attendant',pipeline)
    assert len(pipeline.store.list_documents()) == 1
    assert any('indexed 1 pages' in msg for msg in messages)
    module.render_document_ingestion('doctor',pipeline)
    assert any('already indexed' in msg for msg in messages)


def test_page_limit_and_multi_page_source_numbers():
    from src.data_processing.pdf_loader import PDFLoader
    writer = PdfWriter()
    writer.add_blank_page(width=100,height=100)
    writer.add_page(__import__('pypdf').PdfReader(BytesIO(pdf_bytes())).pages[0])
    buffer = BytesIO()
    writer.write(buffer)
    with pytest.raises(ValueError,match='page limit'):
        PDFLoader.load_pages(buffer.getvalue(),max_pages=1)
    pages = PDFLoader.load_pages(buffer.getvalue())
    assert pages[0]['text'] == ''
    assert pages[1]['page'] == 2 and 'Kidney' in pages[1]['text']
