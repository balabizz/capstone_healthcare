from copy import deepcopy
import json
from unittest.mock import Mock
import pytest
from src.evaluation.runner import Evaluator, aggregate, load_dataset, validate_grade, save_report


def grade():
    return {'accuracy':1,'relevance':1,'faithfulness':1,'hallucination':False,'unsupported_claims':[],
            'rationale':{name:'Supported.' for name in ('accuracy','relevance','faithfulness')}}


def test_dataset_covers_three_production_tasks():
    data,digest = load_dataset('evaluation/healthcare_cases.json')
    assert len(data['cases']) == 9 and len(digest)==64
    assert {c['task'] for c in data['cases']} == {'reference_qa','history_summary','medical_search_summary'}


def test_failures_not_counted_as_good_scores(tmp_path):
    dataset,digest = load_dataset('evaluation/healthcare_cases.json')
    evaluator = Evaluator()
    evaluator.generate = Mock(side_effect=['answer',ValueError('generation failure'),'answer'])
    evaluator.grade = Mock(side_effect=[grade(),ValueError('judge failure')])
    dataset['cases'] = dataset['cases'][:3]
    report = evaluator.run(dataset,digest)
    assert report['summary']['evaluated'] == 1
    assert report['summary']['failed'] == 2
    assert report['summary']['evaluation_completion_rate'] == 1/3
    assert report['summary']['accuracy'] == 1
    assert report['results'][1]['status'] == 'generation_failed'
    assert report['results'][2]['status'] == 'judge_failed'
    save_report(report,tmp_path/'result.json')
    assert json.loads((tmp_path/'result.json').read_text()) == report


def test_empty_aggregate_is_not_perfect_score():
    assert aggregate([])['accuracy'] is None
    assert aggregate([])['hallucination'] is None


@pytest.mark.parametrize('changes',[{'accuracy':2},{'faithfulness':True},{'hallucination':True},
                                  {'unsupported_claims':['invented']},{'rationale':{}}])
def test_invalid_judgments_rejected(changes):
    payload = grade()
    payload.update(changes)
    with pytest.raises(ValueError):
        validate_grade(payload)


def test_generation_uses_production_prompts_and_summary_methods():
    dataset,_ = load_dataset('evaluation/healthcare_cases.json')
    evaluator = Evaluator()
    evaluator.reference_client = Mock()
    evaluator.summary_client = Mock()
    for case in dataset['cases']:
        evaluator.generate(case)
    assert evaluator.reference_client._complete.call_count == 2
    assert evaluator.summary_client.summarize_history.call_count == 3
    assert evaluator.summary_client.summarize_medical_search.call_count == 3


def test_judge_gets_candidate_reference_and_evidence():
    data,_ = load_dataset('evaluation/healthcare_cases.json')
    evaluator = Evaluator()
    evaluator.judge._complete = Mock(return_value=json.dumps(grade()))
    assert evaluator.grade(data['cases'][0],'Candidate') == grade()
    payload = json.loads(evaluator.judge._complete.call_args.args[1])
    assert payload['candidate_answer']=='Candidate' and payload['evidence'] and payload['reference_answer']


def test_judge_controls_identify_hallucination_grade():
    result = grade()
    result.update(accuracy=0,faithfulness=0,hallucination=True,unsupported_claims=['Invented outcome'])
    assert validate_grade(result)['hallucination'] is True


def test_evaluation_view_requires_staff_and_shows_counts(monkeypatch):
    from types import SimpleNamespace
    from contextlib import nullcontext
    import importlib.util
    import sys
    from pathlib import Path
    messages = []
    ui = SimpleNamespace(expander=lambda *a,**k:nullcontext(),caption=messages.append,write=messages.append,
                         json=lambda obj:None,dataframe=lambda *a,**k:None,warning=messages.append,info=messages.append)
    monkeypatch.setitem(sys.modules,'streamlit',ui)
    spec = importlib.util.spec_from_file_location('evaluation_view_test',Path('app/evaluation_view.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    execution = Mock()
    execution._connect.return_value.__enter__ = Mock(return_value=SimpleNamespace(execute=lambda *a:[]))
    execution._connect.return_value.__exit__ = Mock(return_value=False)
    module.render_model_evaluation('patient',execution)
    assert not messages
    import app.performance_view
    monkeypatch.setattr(app.performance_view,'render_quality_charts',lambda report:None)
    module.render_model_evaluation('doctor',execution)
    assert any('Evaluated' in str(m) or 'No evaluation report' in str(m) for m in messages)
