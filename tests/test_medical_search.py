from datetime import datetime, timezone
import json
from unittest.mock import Mock
import pytest
import requests

from src.tools.medical_search import MedicalSearch, ProviderError, safe_topic
from src.llm.medical_search_summary import render_summary, schema_for
from src.llm.planning_client import PlanningClient

WHO = {'value': [{'Id':'who-123','Title':'Diabetes guidance','UrlName':'diabetes-guidance',
                 'PublicationDateAndTime':'2026-01-01T00:00:00Z','Overview':'<p>Published guidance overview.</p><script>bad()</script>'}], '@odata.count':1}


class Response:
    def __init__(self, payload, status=200):
        self.payload = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
        self.status_code = status
        self.closed = False
    def iter_content(self, size):
        yield self.payload
    def close(self):
        self.closed = True


def service(who=None):
    session = Mock()
    session.get.side_effect = [Response(who or WHO)]
    client = MedicalSearch(session=session,now=lambda:datetime(2026,9,20,tzinfo=timezone.utc), throttle=False)
    return client, session


def test_live_search_contract_dates_urls_and_evidence():
    client, session = service()
    result = client.search('diabetes treatment')
    assert result['status'] == 'success'
    assert len(result['sources']) == 1
    who = result['sources'][0]
    assert who['id'] == 'who:who-123'
    assert who['url'] == 'https://www.who.int/publications/i/item/diabetes-guidance'
    assert 'bad()' not in who['excerpt']
    params = session.get.call_args_list[0].kwargs['params']
    assert 'PublicationDateAndTime' in params['$filter']
    assert all(call.kwargs['allow_redirects'] is False for call in session.get.call_args_list)


def test_partial_failure_and_no_local_rag_fallback():
    client, session = service()
    session.get.side_effect = [requests.Timeout()]
    result = client.search('diabetes')
    assert result['status'] == 'failed'
    assert result['sources'] == []
    assert result['providers'][0]['provider'] == 'WHO'


def test_empty_and_total_failure_are_explicit():
    client, session = service()
    session.get.side_effect = [Response({'value':[]})]
    assert client.search('diabetes')['status'] == 'no_results'
    session.get.side_effect = [requests.Timeout()]
    assert client.search('diabetes')['status'] == 'failed'


@pytest.mark.parametrize('query', ['my father kidney disease','patient_id 12345 kidney', 'patient@example.com diabetes',
                                 'https://private.example.com','patient DOB 1990-01-01', 'x'])
def test_obvious_personal_identifiers_never_reach_providers(query):
    client, session = service()
    with pytest.raises(ValueError):
        client.search(query)
    session.get.assert_not_called()


def test_download_limits_and_entity_rejection():
    client, session = service()
    response = Response('x' * 2_000_001)
    session.get.side_effect = [response]
    with pytest.raises(ProviderError):
        client._get('https://www.who.int/api/hubs/publications', {})
    assert response.closed
    client, session = service()
    assert client.search('diabetes')['status'] == 'success'


def test_metadata_only_cannot_support_medical_claims():
    client, session = service(who={'value': [{'Id':'who-123','Title':'Diabetes guidance','UrlName':'diabetes-guidance','PublicationDateAndTime':'2026-01-01T00:00:00Z','Overview':''}], '@odata.count':1})
    result = client.search('diabetes')
    schema = schema_for(result)
    assert schema['properties']['findings']['items']['properties']['sources']['items']['enum'] == []
    with pytest.raises(ValueError):
        render_summary({'findings':[{'text':'Treatment claim','sources':['who:who-123']}]}, result)


def test_citations_are_built_from_provider_urls_not_llm_links():
    client, session = service()
    evidence = client.search('diabetes')
    output = render_summary({'findings':[{'text':'Published guidance is available.', 'sources':['who:who-123']}]}, evidence)
    assert '(https://www.who.int/publications/i/item/diabetes-guidance)' in output
    assert '2026-01-01T00:00:00Z' in output
    for claim in ({'text':'A claim','sources':['who:other']},
                  {'text':'See https://untrusted.example','sources':['who:who-123']}):
        with pytest.raises(ValueError):
            render_summary({'findings':[claim]}, evidence)


def test_openai_summary_receives_excerpts_and_exact_source_enum():
    client, _ = service()
    evidence = client.search('diabetes')
    session = Mock()
    session.post.return_value.json.return_value = {'choices':[{'finish_reason':'stop','message':{'content':json.dumps({
            'findings':[{'text':'The guidance provides an overview.','sources':['who:who-123']}]})}}]}
    llm = PlanningClient(api_key='synthetic',session=session)
    answer = llm.summarize([{'goal':'medical_search','status':'success','search_evidence':evidence,'message':'Found sources'}])
    assert 'https://www.who.int/publications/i/item/diabetes-guidance' in answer
    body = session.post.call_args.kwargs['json']
    assert 'untrusted DATA' in body['messages'][0]['content']
    assert 'Published guidance overview' in body['messages'][1]['content']
    assert body['response_format']['json_schema']['name'] == 'medical_search_summary'


def test_no_usable_evidence_does_not_call_llm():
    session = Mock()
    result = PlanningClient(api_key='test',session=session).summarize_medical_search({'sources':[]})
    assert 'No live abstracts' in result
    session.post.assert_not_called()


def test_request_words_are_removed_before_external_query():
    client, session = service()
    result = client.search('Find latest diabetes treatments')
    assert result['query'] == 'diabetes treatments'
    params = session.get.call_args_list[0].kwargs['params']
    assert 'Find' not in params['$filter'] and 'latest' not in result['query']


def test_executor_exposes_live_sources_and_logs_only_metadata(tmp_path):
    from src.agents.goal_execution import GoalExecution
    from src.agents.plan_execution import PlanExecution
    from src.agents.planner import Planner
    execution = GoalExecution(str(tmp_path / 'search.db'))
    execution.medical_search, _ = service()
    plan = Planner.validate({'relationship':None,'clarification':None,'steps':[
        {'id':'search','name':'medical_search','query':'diabetes treatment','specialty':None,'depends_on':[]},
        {'id':'summary','name':'final_summary','query':'Summarize sources','specialty':None,'depends_on':['search']}]})
    result = PlanExecution(execution, lambda evidence: 'Source-based answer').run(plan,request_id='search-test')
    assert result['medical_search']['sources'][0]['id'] == 'who:who-123'
    assert result['steps'][0]['status'] == 'success'
    assert 'publication window' in result['answer']
    assert 'uncertain results' not in json.dumps(execution.events.list_events(request_id='search-test'))


def test_public_search_synthesis_does_not_receive_patient_history():
    from unittest.mock import patch
    from src.llm.planning_client import PlanningClient
    client, _ = service()
    search = client.search('diabetes')
    llm = PlanningClient(api_key='test-placeholder')
    history = {'private':'patient clinical data'}
    with patch.object(llm, 'summarize_history', return_value='History summary') as history_method, \
         patch.object(llm, 'summarize_medical_search', return_value='Cited search summary') as search_method:
        answer = llm.summarize([{'goal':'history_retrieval','history_snapshot':history},
                               {'goal':'medical_search','search_evidence':search}])
    assert history_method.call_args.args[0] is history
    assert 'patient clinical data' not in json.dumps(search_method.call_args.args[0])
    assert 'History summary' in answer and 'Cited search summary' in answer
