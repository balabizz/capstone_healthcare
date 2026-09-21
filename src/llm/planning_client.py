"""OpenAI HTTP adapter isolated from the legacy LangChain dependencies."""

import json
import requests
from src.config import OPENAI_API_KEY, PLANNER_MODEL
from src.llm.task_prompts import FINAL_SUMMARY_PROMPT


class PlanningClient:
    def __init__(self, api_key=OPENAI_API_KEY, model=PLANNER_MODEL, session=None, temperature=None):
        self.temperature = temperature
        self.api_key = api_key
        self.model = model
        self.session = session or requests.Session()

    def _complete(self, system, user, response_format=None):
        if not self.api_key:
            raise ValueError('Set OPENAI_API_KEY in .env to enable planning.')
        body = {'model': self.model, 'messages': [
            {'role': 'system', 'content': system}, {'role': 'user', 'content': user}]}
        if self.temperature is not None:
            body['temperature'] = self.temperature
        if response_format:
            body['response_format'] = response_format
        try:
            response = self.session.post('https://api.openai.com/v1/chat/completions',
                headers={'Authorization': f'Bearer {self.api_key}'}, json=body, timeout=(10, 60))
            response.raise_for_status()
            choice = response.json()['choices'][0]
            if choice['finish_reason'] != 'stop' or choice['message'].get('refusal'):
                raise ValueError('OpenAI could not complete this request. Please rephrase it.')
            content = choice['message']['content']
            if not isinstance(content, str) or not content.strip():
                raise ValueError('OpenAI returned an empty response.')
            return content
        except (requests.RequestException, KeyError, IndexError, TypeError) as error:
            raise ValueError('OpenAI request failed. Check your API key, planner model and connection.') from error

    def plan(self, system, user, schema):
        content = self._complete(system, user, {'type': 'json_schema', 'json_schema': {
            'name': 'healthcare_plan', 'strict': True, 'schema': schema}})
        try:
            return json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError('OpenAI returned an invalid planning response.') from error

    def summarize_history(self, bundle):
        from src.llm.history_summary import SUMMARY_PROMPT, schema_for, render_summary
        # Exclude the patient identity from the model request; the UI identifies the subject.
        evidence = {k: v for k, v in bundle.items() if k != 'subject_patient_id'}
        content = self._complete(SUMMARY_PROMPT, json.dumps(evidence), {
            'type': 'json_schema', 'json_schema': {'name': 'patient_history_summary',
                                                 'strict': True, 'schema': schema_for(bundle)}})
        try:
            return render_summary(json.loads(content), bundle)
        except json.JSONDecodeError as error:
            raise ValueError('OpenAI returned an invalid patient history summary.') from error

    def summarize_medical_search(self, bundle):
        from src.llm.medical_search_summary import PROMPT, schema_for, usable_sources, render_summary
        if not usable_sources(bundle):
            return 'No live abstracts or publication overviews were available to support a medical answer.'
        evidence = {**bundle, 'sources': usable_sources(bundle)}
        content = self._complete(PROMPT, json.dumps(evidence), {
            'type': 'json_schema', 'json_schema': {'name': 'medical_search_summary',
                                                'strict': True, 'schema': schema_for(bundle)}})
        try:
            return render_summary(json.loads(content), bundle)
        except json.JSONDecodeError as error:
            raise ValueError('OpenAI returned an invalid medical-search summary.') from error

    def summarize(self, evidence):
        self.last_history_summary = None
        history = next((r['history_snapshot'] for r in evidence if 'history_snapshot' in r), None)
        search = next((r['search_evidence'] for r in evidence if 'search_evidence' in r), None)
        memory = next((r['conversation_memory'] for r in evidence if 'conversation_memory' in r), None)
        if history is not None or search is not None or memory is not None:
            parts = []
            if memory is not None:
                parts.append('Past conversation excerpts (historical dialogue, not verified current clinical facts):\n' +
                    ('\n\n'.join(f"{r['created_at']} [turn:{r['turn_id']}]\nYou: {r['query']}\nAssistant: {r['answer']}" for r in memory)
                     if memory else 'No saved conversations were found for this patient and account.'))
            other = [r for r in evidence if r['goal'] not in ('patient_lookup', 'history_retrieval', 'medical_search', 'conversation_recall')]
            if other:
                if len(other) == 1 and other[0]['goal'] == 'medical_question':
                    parts.append(other[0]['message'])
                else:
                    parts.append('Other requested tasks:\n' + '\n'.join(
                        f"- {r['goal'].replace('_', ' ')} ({r['status']}): {r['message']}" for r in other))
            if history is not None:
                has_history_entries = any(history.get(key) for key in ('records', 'prescriptions', 'alerts'))
                if not has_history_entries:
                    if {'retrieved_at', 'coverage'}.issubset(history):
                        from src.llm.history_summary import coverage_notice
                        parts.append('No clinical history entries were recorded in the authorized snapshot.\n' + coverage_notice(history))
                    else:
                        parts.append('No clinical history summary was available for this request.')
                else:
                    try:
                        self.last_history_summary = self.summarize_history(history)
                        parts.append(self.last_history_summary)
                    except Exception:
                        from src.llm.history_summary import coverage_notice
                        parts.append('Patient history was retrieved but could not be summarized by the model. '
                                     'Review the authorized history source records.\n' + coverage_notice(history))
            if search is not None:
                parts.append('Current medical publication search:\n' + self.summarize_medical_search(search))
            return '\n\n'.join(parts)
        if any(r['goal'] == 'medical_question' and r['status'] == 'success' for r in evidence):
            # Preserve the reference answer: a second model has no retrieved excerpts
            # with which to verify a rewritten clinical claim.
            return '\n\n'.join(r['message'] for r in evidence if r['goal'] != 'patient_lookup')
        return self._complete(FINAL_SUMMARY_PROMPT, json.dumps(evidence))
