"""Isolated synthetic planner/executor exercises; never use the production database."""
from tempfile import TemporaryDirectory
from pathlib import Path
from src.agents.goal_execution import GoalExecution
from src.agents.conversation_flow import ConversationFlow
from src.agents.plan_execution import PlanExecution
from src.agents.planner import Planner
from src.models.patient_vo import PatientVO
from src.models.doctor_vo import DoctorVO

SCENARIOS = {
    'Father appointment': {'query':'Find available nephrologist slots for my father so I can choose one.',
                           'expected':['patient_lookup','specialist_discovery','appointment'], 'grant':True},
    'Father history denied': {'query':"Summarize my father's saved medical history.",
                              'expected':['patient_lookup','history_retrieval'], 'grant':False},
    'Remembered follow-up': {'query':'What are its complications?', 'expected':['medical_question'],'grant':True},
    'Recall conversation': {'query':'What did we discuss previously?', 'expected':['patient_lookup','conversation_recall'],'grant':True},
}


def run_scenario(name, query=None, planner=None):
    scenario = SCENARIOS[name]
    with TemporaryDirectory(prefix='healthcare-scenario-') as directory:
        execution = GoalExecution(str(Path(directory)/'synthetic.db'))
        execution.schedule = execution.calendars  # Never inherit the configured remote schedule backend.
        store = execution.patient_history.store
        for patient in ('demo-requester','demo-father'):
            store.save_patient(PatientVO(patient_id=patient,first_name='Synthetic',last_name='Patient'))
        store.save_doctor(DoctorVO(doctor_id='demo-doctor',first_name='Demo',last_name='Doctor',speciality='Nephrology'))
        execution.dependents.add_dependent('demo-requester','Synthetic father','father','demo-father')
        execution.dependents.set_permission('demo-father','demo-requester','book_appointment',True)
        execution.dependents.set_permission('demo-father','demo-requester','view_medical',scenario['grant'])
        for day in range(7):
            execution.calendars.set_hours('demo-doctor',day,'09:00','17:00')
        execution.conversations.save('demo-requester','demo-requester','Tell me about diabetes',
                                     'We discussed general diabetes information; no personal diagnosis was made.','synthetic-turn')
        # Fixture adapters are explicitly disclosed in the interface/result.
        def reference(question,**kwargs):
            if kwargs.get('check_access'):
                kwargs['check_access']()
            return {'answer':'Synthetic reference response: the medical answer model was not invoked.', 'source_documents':[]}
        execution.answer_question = reference
        class NoLiveSearch:
            def search(self,query):
                return {'status':'no_results','query':query,'sources':[],'providers':[],
                        'searched_at':'synthetic scenario','date_from':'synthetic','date_to':'synthetic'}
        execution.medical_search = NoLiveSearch()
        flow = ConversationFlow(execution,planner or Planner())
        plan,subject = flow.plan(query or scenario['query'],requester='demo-requester')
        result = PlanExecution(execution,lambda rows:'\n'.join(f"{r['goal']}: {r['status']}" for r in rows)).run(plan,patient_id='demo-requester')
        with execution._connect() as c:
            bookings = c.execute('SELECT COUNT(*) FROM appointments').fetchone()[0]
        actual = [g.name for g in plan.goals]
        checks = {'expected_subgoals_present':all(g in actual for g in scenario['expected']), 'no_booking_written':bookings==0}
        if name=='Father history denied':
            checks['history_access_denied'] = any(r['goal']=='history_retrieval' and r['status']=='denied' for r in result['steps'])
        if name=='Father appointment':
            checks['awaits_confirmation'] = any(r['goal']=='appointment' and r['status']=='awaiting_confirmation' for r in result['steps'])
        return {'checks':checks,'scenario':name,'mode':'Synthetic database; live OpenAI planner; fixture reference answers and final summary; no external medical search.',
                'plan':plan.details(),'steps':result['steps'],
                'memory':flow.memory_traces+result.get('memory_traces',[]),
                'expected_goals':scenario['expected'],'missing_expected_goals':[g for g in scenario['expected'] if g not in actual],
                'clarification':plan.clarification,'synthetic_bookings_written':bookings}
