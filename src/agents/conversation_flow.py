"""Scope memory using account selection and validated patient resolution."""
from src.agents.memory_trace import memory_trace


class ConversationFlow:
    def __init__(self, execution, planner):
        self.execution, self.planner = execution, planner
        self.memory_traces = []

    def subject(self, requester, dependent_id=None, relationship=None):
        return (self.execution.dependents.resolve(requester, relationship, dependent_id)
                if dependent_id or relationship else requester)

    def plan(self, query, *, requester, dependent_id=None):
        self.memory_traces = []
        # Resolve explicit relationships before exposing any saved dialogue to the planner.
        initial = self.planner.plan(query, selected_family=bool(dependent_id))
        if not requester:
            return initial, None
        subject = self.subject(requester, dependent_id, initial.relationship)
        try:
            context = self.execution.conversations.retrieve(requester, subject, query)
        except PermissionError:
            self.memory_traces.append({'stage':'planner','status':'denied','count':0,'turns':[]})
            return initial, subject
        self.memory_traces.append(memory_trace('planner',context))
        if not context:
            return initial, subject
        plan = self.planner.plan(query, selected_family=bool(dependent_id), conversation_context=context)
        if self.subject(requester, dependent_id, plan.relationship) != subject:
            self.memory_traces[-1]['status'] = 'discarded_subject_change'
            # Never transfer remembered preferences/clinical statements to another subject.
            return initial, subject
        self.execution.patient_history.require_access(requester, subject)
        return plan, subject

    def remember(self, query, result, *, requester, subject, request_id):
        if not requester or not subject:
            return False
        if any(s['status'] == 'denied' for s in result['steps']):
            return False
        self.execution.conversations.save(requester, subject, query, result['answer'], request_id)
        return True
