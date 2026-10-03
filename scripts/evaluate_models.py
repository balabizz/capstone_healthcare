"""Run paid OpenAI generation/judging on the synthetic fixed-evidence benchmark."""
import argparse
import json
from src.config import PROJECT_ROOT, LLM_MODEL, PLANNER_MODEL
from src.evaluation.runner import Evaluator, load_dataset, save_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',default=str(PROJECT_ROOT/'evaluation/healthcare_cases.json'))
    parser.add_argument('--output',default=str(PROJECT_ROOT/'reports/evaluation/latest.json'))
    parser.add_argument('--answer-model',default=LLM_MODEL)
    parser.add_argument('--summary-model',default=PLANNER_MODEL)
    parser.add_argument('--judge-model',default=PLANNER_MODEL)
    args = parser.parse_args(argv)
    dataset,digest = load_dataset(args.dataset)
    evaluator = Evaluator(args.answer_model,args.summary_model,args.judge_model)
    report = evaluator.run(dataset,digest,progress=lambda id,status:print(f'{id}: {status}',flush=True))
    # Known-good and known-hallucinated answers test judge discrimination, not model quality.
    control = {'id':'judge_control','question':'When does the synthetic desk open?',
               'evidence':{'context':'The fictional desk opens at 09:00.'},
               'reference_answer':'09:00.','rubric':'Only 09:00 is supported.'}
    report['judge_controls'] = []
    for label,answer,expected in [('supported','The desk opens at 09:00.',False),
                                  ('unsupported','The desk opens at 17:00 and provides free medicine.',True)]:
        try:
            grade = evaluator.grade(control,answer)
            report['judge_controls'].append({'id':label,'answer':answer,'grade':grade,
                'passed':grade['hallucination'] == expected and (expected or grade['accuracy']==1)})
        except Exception as error:
            report['judge_controls'].append({'id':label,'passed':False,'error_type':type(error).__name__})
    save_report(report,args.output)
    print(json.dumps(report['summary'],indent=2))
    print(f'Report saved to {args.output}')
    return 1 if report['summary']['failed'] or not all(c['passed'] for c in report['judge_controls']) else 0


if __name__ == '__main__':
    raise SystemExit(main())
