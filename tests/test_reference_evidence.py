from unittest.mock import Mock
from contextlib import nullcontext
from types import SimpleNamespace
import importlib.util
from pathlib import Path
import sys
from langchain_core.documents import Document
from tests.test_dependents import service
from tests.test_medical_question_context import medical_plan
from src.agents.plan_execution import PlanExecution
from src.llm.planning_client import PlanningClient
from src.llm.reference_evidence import reference_evidence, reference_notice


def test_sources_survive_execution_and_appear_in_final_answer(service):
    doc = Document(page_content='Source excerpt about complications.',metadata={'source':'guideline.pdf','page':3,'chunk_index':2})
    service.answer_question = Mock(return_value={'answer':'Reference-grounded answer.','source_documents':[doc]})
    client = PlanningClient()
    client._complete = Mock(side_effect=AssertionError('Must not rewrite the reference answer'))
    result = PlanExecution(service,client.summarize).run(medical_plan(),patient_id='caller')
    assert result['source_documents'] == [doc]
    assert result['reference_evidence'][0]['excerpt'] == doc.page_content
    assert result['answer'].startswith('Reference-grounded answer.')
    assert '[ref:1] guideline.pdf — page 3' in result['answer']
    client._complete.assert_not_called()


def test_no_sources_produces_no_fabricated_reference_labels(service):
    service.answer_question = Mock(return_value={'answer':'No evidence available.','source_documents':[]})
    result = PlanExecution(service,PlanningClient().summarize).run(medical_plan(),patient_id='caller')
    assert result['reference_evidence'] == []
    assert 'ref:' not in result['answer']


def test_missing_metadata_and_multiple_chunks_are_honest():
    docs = [Document(page_content='First'),Document(page_content='Second',metadata={'source':'guide.pdf','page':2}),
            Document(page_content='Third',metadata={'source':'guide.pdf','page':2})]
    rows = reference_evidence(docs)
    assert [r['id'] for r in rows] == ['ref:1','ref:2','ref:3']
    assert 'Unknown source — page not supplied' in reference_notice(rows)
    assert [r['excerpt'] for r in rows] == ['First','Second','Third']


def test_revoked_context_removes_answer_sources_and_evidence(service):
    dependent = service.dependents.add_dependent('caller','Dad','father','father')
    service.dependents.set_permission('father','caller','view_medical',True)
    service.answer_question = Mock(return_value={'answer':'Private answer','source_documents':[
        Document(page_content='Private excerpt',metadata={'source':'Private source'})]})
    def final(evidence):
        service.dependents.set_permission('father','caller','view_medical',False)
        return 'Private rewritten answer'
    result = PlanExecution(service,final).run(medical_plan(True),patient_id='caller',dependent_id=dependent)
    assert result['source_documents'] == result['reference_evidence'] == []
    assert 'Private' not in str(result)


def test_source_panel_labels_and_displays_full_excerpts(monkeypatch):
    rendered = []
    ui = SimpleNamespace(expander=lambda *a,**k:nullcontext(),caption=lambda text:None,
                         text=rendered.append,info=rendered.append)
    monkeypatch.setitem(sys.modules,'streamlit',ui)
    spec = importlib.util.spec_from_file_location('source_view_test',Path('app/reference_evidence_view.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.render_reference_evidence([Document(page_content='Exact excerpt',metadata={'source':'guide.pdf','page':4})])
    assert rendered == ['[ref:1] guide.pdf · page 4','Exact excerpt']


def test_filename_cannot_create_markdown_links_in_answer():
    rows = reference_evidence([Document(page_content='Text',metadata={'source':'[fake](https://example.org)\n# pretend'})])
    notice = reference_notice(rows)
    assert '[fake]' not in notice
    assert '\n# pretend' not in notice
