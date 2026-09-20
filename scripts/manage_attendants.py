"""Trusted local administration of attendant accounts and patient assignments.

Examples:
python -m scripts.manage_attendants create --username attendant.one --first-name Demo --last-name Attendant --patient patient-001
python -m scripts.manage_attendants assign --username attendant.one --patient patient-002
python -m scripts.manage_attendants revoke --username attendant.one --patient patient-002
python -m scripts.manage_attendants disable --username attendant.one
Passwords are entered at a hidden prompt, never as command-line arguments.
"""
import argparse
import sqlite3
from getpass import getpass
from uuid import uuid4
from src.config import SQLITE_DB_PATH
from src.database.sqlite_store import SQLiteStore
from src.utils.passwords import hash_password


def create_attendant(database_path, username, first_name, last_name, password, patient_ids):
    if not username.strip() or not first_name.strip() or not last_name.strip():
        raise ValueError('Username and attendant name are required.')
    store = SQLiteStore(database_path)
    password_hash = hash_password(password)
    attendant_id = f'attendant-{uuid4().hex}'
    with store._connect() as c:
        c.execute('INSERT INTO attendants VALUES (?,?,?)', (attendant_id, first_name.strip(), last_name.strip()))
        c.execute("INSERT INTO login_details (login_id,user_type,attendant_id,username,password_hash) VALUES (?,'attendant',?,?,?)",
                  (uuid4().hex, attendant_id, username.strip().lower(), password_hash))
        c.executemany('INSERT INTO attendant_patients VALUES (?,?)', [(attendant_id, p) for p in set(patient_ids)])
    return attendant_id


def manage_access(database_path, username, action, patient_ids=()):
    store = SQLiteStore(database_path)
    with store._connect() as c:
        row = c.execute("SELECT attendant_id FROM login_details WHERE username=? AND user_type='attendant'",
                        (username.strip().lower(),)).fetchone()
        if not row:
            raise ValueError('Attendant account not found.')
        if action in ('disable', 'enable'):
            c.execute('UPDATE login_details SET is_active=?, updated_at=CURRENT_TIMESTAMP WHERE attendant_id=?',
                      (int(action == 'enable'), row['attendant_id']))
        elif action == 'assign':
            c.executemany('INSERT OR IGNORE INTO attendant_patients VALUES (?,?)', [(row['attendant_id'], p) for p in set(patient_ids)])
        elif action == 'revoke':
            c.executemany('DELETE FROM attendant_patients WHERE attendant_id=? AND patient_id=?', [(row['attendant_id'], p) for p in set(patient_ids)])
        else:
            raise ValueError('Unsupported administrative action.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', default=SQLITE_DB_PATH)
    subs = parser.add_subparsers(dest='action', required=True)
    for action in ('create', 'assign', 'revoke', 'disable', 'enable'):
        sub = subs.add_parser(action)
        sub.add_argument('--username', required=True)
        if action in ('create', 'assign', 'revoke'):
            sub.add_argument('--patient', action='append', required=True)
        if action == 'create':
            sub.add_argument('--first-name', required=True)
            sub.add_argument('--last-name', required=True)
    args = parser.parse_args()
    try:
        if args.action == 'create':
            password = getpass('New attendant password (at least 12 characters): ')
            if getpass('Confirm password: ') != password:
                raise ValueError('Passwords do not match.')
            create_attendant(args.database, args.username, args.first_name, args.last_name, password, args.patient)
        else:
            manage_access(args.database, args.username, args.action, getattr(args, 'patient', ()))
        print('Attendant account/access updated.')
    except (ValueError, sqlite3.IntegrityError) as error:
        parser.exit(1, f'Unable to update attendant: {error}\n')
