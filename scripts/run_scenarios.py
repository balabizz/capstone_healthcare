"""Exercise isolated synthetic scenarios with the configured OpenAI planner."""
import argparse
from datetime import datetime, timezone
from src.agents.scenario_testing import SCENARIOS, run_scenario
from src.agents.planner import Planner
from src.llm.planning_client import PlanningClient
from src.evaluation.runner import save_report
from src.config import PROJECT_ROOT, PLANNER_MODEL


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario',choices=['all',*SCENARIOS],default='all')
    parser.add_argument('--output',default=str(PROJECT_ROOT/'reports/scenarios/latest.json'))
    args=parser.parse_args(argv)
    results=[]
    class RecordedClient(PlanningClient):
        def __init__(self):
            super().__init__()
            self.attempts=[]
        def plan(self,*args,**kwargs):
            payload=super().plan(*args,**kwargs)
            self.attempts.append(payload)
            return payload
    for name in SCENARIOS if args.scenario=='all' else [args.scenario]:
        client=RecordedClient()
        try:
            result=run_scenario(name,planner=Planner(client))
            result['status']='passed' if all(result['checks'].values()) else 'checks_failed'
        except Exception as error:
            result={'scenario':name,'status':'failed','error_type':type(error).__name__,
                    'error':str(error) if isinstance(error,(ValueError,PermissionError)) else 'Scenario execution failed.'}
        result['planning_attempts']=client.attempts
        results.append(result)
        print(f"{name}: {result['status']}",flush=True)
    save_report({'run_at':datetime.now(timezone.utc).isoformat(),'planner_model':PLANNER_MODEL,
                 'scope':'Live planning, real executor and local repositories; synthetic temporary data and fixture medical responses.',
                 'results':results},args.output)
    return 0 if all(r['status']=='passed' for r in results) else 1


if __name__=='__main__':
    raise SystemExit(main())
