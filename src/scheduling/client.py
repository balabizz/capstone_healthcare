"""Private Doctor Schedule API adapter using the same local service contract."""
import requests


class ScheduleUnavailable(ValueError):
    """Transport failure; confirmation outcome may be unknown until idempotent retry."""


class ScheduleAPIClient:
    def __init__(self, url, token, session=None):
        if not token:
            raise ValueError('SCHEDULE_API_TOKEN is required for the Schedule API.')
        self.url, self.token = url.rstrip('/'), token
        self.session = session or requests.Session()

    def _post(self, path, payload):
        try:
            response = self.session.post(self.url + path, json=payload,
                headers={'Authorization': 'Bearer ' + self.token}, timeout=(5, 20))
            if response.status_code == 403:
                raise PermissionError('The Schedule API denied this operation.')
            if response.status_code == 400:
                raise ValueError('The Schedule API rejected these preferences. Check the date range and booking details.')
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, requests.exceptions.JSONDecodeError) as error:
            raise ScheduleUnavailable('Schedule API unavailable. If confirming a booking, retry the same request; '
                             'its request key prevents duplicate appointments.') from error

    def specialists(self, specialty=None, doctor_name=None):
        return self._post('/specialists', {'specialty': specialty, 'doctor_name': doctor_name})

    def discover(self, **preferences):
        return self._post('/slots', preferences)

    def book(self, **booking):
        return self._post('/bookings', booking)
