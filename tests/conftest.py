"""Unit tests must not make paid or external network requests."""
import pytest


@pytest.fixture(autouse=True)
def prohibit_external_network(monkeypatch):
    import requests
    def denied(*args, **kwargs):
        raise AssertionError('Network calls are disabled in unit tests; inject a fake transport.')
    monkeypatch.setattr(requests.Session, 'request', denied)
