"""Explicitly add demo doctor calendars; never overwrite existing calendars."""
import argparse
from src.config import SQLITE_DB_PATH
from src.repositories.schedule_repository import ScheduleRepository


def seed_calendars(database_path=SQLITE_DB_PATH):
    calendars = ScheduleRepository(database_path)
    count = 0
    for index, doctor in enumerate(calendars.specialists()):
        if calendars.list_hours(doctor['doctor_id']):
            continue
        # Different doctors work different days and sessions; lunch stays unavailable.
        weekdays = (0, 2, 4) if index % 2 == 0 else (1, 3, 5)
        for weekday in weekdays:
            calendars.set_hours(doctor['doctor_id'], weekday, '09:00', '12:00')
            calendars.set_hours(doctor['doctor_id'], weekday, '14:00', '17:00')
        count += 1
    return count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', default=SQLITE_DB_PATH)
    args = parser.parse_args()
    print(f'Added demo working calendars for {seed_calendars(args.database)} doctors.')
