"""Doctor calendars, deterministic discovery and atomic 30-minute bookings."""
from datetime import date, datetime, time, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo
import json

from src.config import SCHEDULE_TIMEZONE
from src.database.sqlite_store import SQLiteStore
from src.repositories.dependent_repository import DependentRepository

ALIASES = {'nephrologist': 'nephrology', 'cardiologist': 'cardiology',
           'dermatologist': 'dermatology', 'pediatrician': 'pediatrics',
           'neurologist': 'neurology', 'urologist': 'urology',
           'psychiatrist': 'psychiatry', 'oncologist': 'oncology',
           'gastroenterologist': 'gastroenterology', 'endocrinologist': 'endocrinology',
           'pulmonologist': 'pulmonology', 'orthopedist': 'orthopedics'}


def specialty_key(value):
    value = (value or '').strip().lower()
    return ALIASES.get(value, value)


def minute(value):
    if not isinstance(value, str) or len(value) != 5:
        raise ValueError('Time must use HH:MM format.')
    parsed = time.fromisoformat(value)
    return parsed.hour * 60 + parsed.minute


class ScheduleRepository:
    def __init__(self, database_path, now=None):
        self.store = SQLiteStore(database_path)
        self.permissions = DependentRepository(database_path)
        self.zone = ZoneInfo(SCHEDULE_TIMEZONE)
        self.now = now or (lambda: datetime.now(self.zone))

    def set_hours(self, doctor_id, weekday, start, end):
        """Trusted doctor-session boundary; add one recurring working window."""
        if type(weekday) is not int or weekday not in range(7) or minute(end) - minute(start) < 30:
            raise ValueError('Choose a weekday and a working window of at least 30 minutes.')
        with self.store._connect() as connection:
            connection.execute('INSERT OR IGNORE INTO doctor_working_hours '
                '(doctor_id, weekday, start_time, end_time) VALUES (?, ?, ?, ?)',
                (doctor_id, weekday, start, end))

    def list_hours(self, doctor_id):
        with self.store._connect() as connection:
            return [dict(r) for r in connection.execute('SELECT * FROM doctor_working_hours '
                'WHERE doctor_id = ? ORDER BY weekday, start_time', (doctor_id,))]

    def remove_hours(self, doctor_id, window_id):
        with self.store._connect() as connection:
            connection.execute('DELETE FROM doctor_working_hours WHERE doctor_id = ? AND window_id = ?',
                               (doctor_id, window_id))

    def set_day_off(self, doctor_id, day, enabled=True):
        date.fromisoformat(day)
        with self.store._connect() as connection:
            if enabled:
                connection.execute('INSERT OR IGNORE INTO doctor_days_off VALUES (?, ?)', (doctor_id, day))
            else:
                connection.execute('DELETE FROM doctor_days_off WHERE doctor_id = ? AND day = ?', (doctor_id, day))

    def list_days_off(self, doctor_id):
        with self.store._connect() as connection:
            return [row['day'] for row in connection.execute(
                'SELECT day FROM doctor_days_off WHERE doctor_id = ? ORDER BY day', (doctor_id,))]

    def specialists(self, specialty=None, doctor_name=None):
        with self.store._connect() as connection:
            rows = connection.execute('SELECT doctor_id, first_name, last_name, speciality FROM doctors '
                                      'ORDER BY last_name, first_name, doctor_id').fetchall()
        # Exact full-name match: never silently choose a different named doctor.
        name = (doctor_name or '').strip().lower().removeprefix('dr. ').removeprefix('dr ')
        return [dict(r) for r in rows if (not specialty or specialty_key(r['speciality']) == specialty_key(specialty))
                and (not name or name == f"{r['first_name']} {r['last_name']}".lower())]

    def _allowed(self, connection, doctor_id, patient_id, day, slot):
        start = minute(slot)
        dt = datetime.combine(day, time.fromisoformat(slot), self.zone)
        # Exclude ambiguous/nonexistent local times around DST transitions.
        if dt.utcoffset() != dt.replace(fold=1).utcoffset() or dt <= self.now():
            return False
        if connection.execute('SELECT 1 FROM doctor_days_off WHERE doctor_id = ? AND day = ?',
                              (doctor_id, day.isoformat())).fetchone():
            return False
        windows = connection.execute('SELECT start_time, end_time FROM doctor_working_hours '
            'WHERE doctor_id = ? AND weekday = ?', (doctor_id, day.weekday())).fetchall()
        if not any(minute(w['start_time']) <= start and start + 30 <= minute(w['end_time'])
                   and (start - minute(w['start_time'])) % 30 == 0 for w in windows):
            return False
        stamp = f'{day.isoformat()} {slot}'
        return not connection.execute("SELECT 1 FROM appointments WHERE status = 'scheduled' "
            "AND (doctor_id = ? OR patient_id = ?) "
            "AND datetime(appointment_datetime) < datetime(?, '+30 minutes') "
            "AND datetime(appointment_datetime, '+30 minutes') > datetime(?)",
            (doctor_id, patient_id, stamp, stamp)).fetchone()

    def discover(self, *, requester_patient_id, patient_id, specialty=None, doctor_name=None,
                 date_from=None, date_to=None, time_from=None, time_to=None, limit=5):
        self.permissions.require_access(requester_patient_id, patient_id, 'book_appointment')
        if not self.store.get_patient(patient_id):
            raise ValueError('Patient does not exist.')
        today = self.now().date()
        first = date.fromisoformat(date_from) if date_from else today
        last = date.fromisoformat(date_to) if date_to else first + timedelta(days=29)
        if first < today or last < first or (last - first).days > 89 or first > today + timedelta(days=365):
            raise ValueError('Choose a future date range of at most 90 days within the next year.')
        earliest = minute(time_from) if time_from else 0
        latest = minute(time_to) if time_to else 24 * 60
        if earliest >= latest or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('Invalid time window or result limit.')
        doctors, result = self.specialists(specialty, doctor_name), []
        with self.store._connect() as connection:
            for offset in range((last - first).days + 1):
                day = first + timedelta(days=offset)
                daily = []
                for doctor in doctors:
                    windows = connection.execute('SELECT start_time, end_time FROM doctor_working_hours '
                        'WHERE doctor_id = ? AND weekday = ?', (doctor['doctor_id'], day.weekday())).fetchall()
                    starts = sorted({n for w in windows for n in range(minute(w['start_time']), minute(w['end_time']) - 29, 30)})
                    for n in starts:
                        slot = f'{n // 60:02d}:{n % 60:02d}'
                        if earliest <= n and n + 30 <= latest and self._allowed(connection, doctor['doctor_id'], patient_id, day, slot):
                            daily.append({**doctor, 'date': day.isoformat(), 'time': slot,
                                          'timezone': str(self.zone), 'duration_minutes': 30})
                result.extend(sorted(daily, key=lambda r: (r['time'], r['doctor_id'])))
                if len(result) >= limit:
                    break
        return result[:limit]

    def book(self, *, requester_patient_id, patient_id, doctor_id, day, slot, reason='', idempotency_key):
        if not isinstance(idempotency_key, str) or not 1 <= len(idempotency_key) <= 128:
            raise ValueError('A booking request key is required.')
        requested = date.fromisoformat(day)
        minute(slot)
        payload = json.dumps([patient_id, doctor_id, day, slot, reason], separators=(',', ':'))
        with self.store._connect() as connection:
            connection.execute('BEGIN IMMEDIATE')
            # Permission rechecked inside the booking lock; revocation cannot race the write.
            if requester_patient_id != patient_id and not connection.execute(
                "SELECT 1 FROM dependent_permissions WHERE subject_patient_id = ? AND requester_patient_id = ? "
                "AND permission = 'book_appointment'", (patient_id, requester_patient_id)).fetchone():
                raise PermissionError('This patient has not granted booking access.')
            prior = connection.execute('SELECT payload, appointment_id FROM booking_requests '
                'WHERE requester_patient_id = ? AND request_key = ?', (requester_patient_id, idempotency_key)).fetchone()
            if prior:
                if prior['payload'] != payload:
                    raise ValueError('This request key was already used for a different booking.')
                return {'status': 'booked', 'appointment_id': prior['appointment_id'], 'replayed': True}
            if not self._allowed(connection, doctor_id, patient_id, requested, slot):
                return {'status': 'slot_unavailable'}
            appointment_id = f'appointment-{uuid4().hex}'
            connection.execute('INSERT INTO appointments (appointment_id, patient_id, doctor_id, appointment_datetime, reason) '
                'VALUES (?, ?, ?, ?, ?)', (appointment_id, patient_id, doctor_id, f'{day} {slot}', reason))
            connection.execute('INSERT INTO booking_requests VALUES (?, ?, ?, ?)',
                (requester_patient_id, idempotency_key, payload, appointment_id))
        return {'status': 'booked', 'appointment_id': appointment_id, 'replayed': False}
