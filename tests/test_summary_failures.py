import json
import unittest
from unittest.mock import Mock

import requests

from src.llm.planning_client import PlanningClient, SummaryFailure


class SummaryFailureTests(unittest.TestCase):
    def setUp(self):
        self.bundle = {
            'records': [{'history_id': 'test-note', 'record_type': 'note',
                         'notes': 'Fabricated note', 'condition_name': 'Test'}],
            'prescriptions': [], 'alerts': [], 'retrieved_at': '2030-01-01', 'coverage': {}}
        self.valid = {'diagnoses_and_treatments': [],
                      'clinical_notes': [{'text': 'Fabricated note', 'sources': ['history:test-note']}],
                      'prescriptions': [], 'alerts': []}

    def test_invalid_source_retried_then_validated(self):
        client = PlanningClient(api_key='test')
        invalid = dict(self.valid, clinical_notes=[{'text': 'Test', 'sources': ['history:wrong']}])
        client._complete = Mock(side_effect=[json.dumps(invalid), json.dumps(self.valid)])
        result = client.summarize_history(self.bundle)
        self.assertIn('history:test-note', result)
        self.assertEqual(client._complete.call_count, 2)
        self.assertIn('record_type=note', client._complete.call_args.args[0])

    def test_repeated_invalid_summary_is_not_published_or_indexed(self):
        client = PlanningClient(api_key='test')
        client._complete = Mock(return_value='not JSON')
        result = client.summarize([{'goal': 'history_retrieval', 'history_snapshot': self.bundle}])
        self.assertIn('[summary:summary_validation]', result)
        self.assertIsNone(client.last_history_summary)
        self.assertEqual(client._complete.call_count, 2)
        self.assertNotIn('not JSON', result)

    def test_network_failure_is_specific_and_not_retried_as_validation(self):
        session = Mock()
        session.post.side_effect = requests.Timeout('PRIVATE KEY OR PATIENT CONTENT')
        client = PlanningClient(api_key='test', session=session)
        result = client.summarize([{'goal': 'history_retrieval', 'history_snapshot': self.bundle}])
        self.assertIn('[summary:timeout]', result)
        self.assertNotIn('PRIVATE', result)
        self.assertEqual(session.post.call_count, 1)

    def test_authentication_and_quota_have_safe_diagnostics(self):
        for status, code in ((401, 'authentication'), (429, 'rate_limit'), (400, 'request_rejected')):
            with self.subTest(status=status):
                response = requests.Response()
                response.status_code = status
                session = Mock()
                session.post.return_value.raise_for_status.side_effect = requests.HTTPError(
                    'PRIVATE PROVIDER RESPONSE', response=response)
                client = PlanningClient(api_key='test', session=session)
                with self.assertRaises(SummaryFailure) as caught:
                    client.summarize_history(self.bundle)
                self.assertEqual(caught.exception.code, code)
                self.assertNotIn('PRIVATE', str(caught.exception))

    def test_unknown_errors_do_not_expose_sensitive_details(self):
        client = PlanningClient(api_key='test')
        client.summarize_history = Mock(side_effect=RuntimeError('PRIVATE RECORD'))
        result = client.summarize([{'goal': 'history_retrieval', 'history_snapshot': self.bundle}])
        self.assertIn('[summary:internal_error]', result)
        self.assertNotIn('PRIVATE', result)
