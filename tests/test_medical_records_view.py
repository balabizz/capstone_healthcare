"""Exercise real editor callbacks with a minimal UI double, not a browser test."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import pytest
from tests.test_medical_records import clinic, save


class Rerun(BaseException):
    pass


class SessionState(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class EditorUI:
    def __init__(self):
        self.session_state = SessionState()
        self.submitted = False
        self.reload = False
        self.selected = 'new_diagnosis'
        self.errors = []

    def __getattr__(self, name):
        if name in ('header', 'caption', 'info', 'write', 'dataframe', 'success', 'json'):
            return lambda *args, **kwargs: None
        raise AttributeError(name)

    def selectbox(self, label, options, **kwargs):
        return self.selected if label == 'Action / record' else options[0]

    def text_input(self, label, value='', **kwargs):
        return {'Diagnosis / condition': 'Documented diagnosis',
                'Reason for this entry or correction': 'Documented correction'}.get(label, value)

    def text_area(self, label, value='', **kwargs):
        return 'Entered clinical text'

    def form(self, *args, **kwargs):
        return nullcontext()

    def expander(self, *args, **kwargs):
        return nullcontext()

    def form_submit_button(self, label, **kwargs):
        return self.submitted and label == 'Save medical record'

    def button(self, *args, **kwargs):
        return self.reload

    def error(self, message):
        self.errors.append(message)

    def rerun(self):
        raise Rerun()


def test_editor_creates_and_retains_stale_version_until_reload(clinic, monkeypatch):
    service, actor = clinic
    ui = EditorUI()
    monkeypatch.setitem(sys.modules, 'streamlit', ui)
    path = Path(__file__).parents[1] / 'app' / 'medical_records_view.py'
    spec = importlib.util.spec_from_file_location('editor_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    profile = {'attendant_id': actor}
    ui.submitted = True
    with pytest.raises(Rerun):
        module.render_medical_records(profile, service)
    original = service.medical_records.list_records(actor, 'patient')[0]
    assert original['condition_name'] == 'Documented diagnosis'
    ui.selected, ui.submitted = original['history_id'], False
    module.render_medical_records(profile, service)  # capture version 1 in the editor
    save(service, actor, history_id=original['history_id'], expected_version=1, notes='Other editor saved')
    ui.submitted = True
    module.render_medical_records(profile, service)
    assert 'changed since you opened' in ui.errors[-1]
    assert ui.session_state['record_editor']['snapshot']['version'] == 1
    ui.reload = True
    with pytest.raises(Rerun):
        module.render_medical_records(profile, service)
    ui.reload = False
    with pytest.raises(Rerun):
        module.render_medical_records(profile, service)
    assert service.medical_records.list_records(actor, 'patient')[0]['version'] == 3
