"""Import patient demographics and summaries from the supplied XLSX dataset.

Usage:
    python -m scripts.import_patient_records
"""

import argparse
import re
import sqlite3
import uuid
from pathlib import Path

import pandas as pd

from src.config import SQLITE_DB_PATH
from src.database.sqlite_store import SQLiteStore
from src.models.patient_vo import PatientVO
from src.repositories.patient_history_repository import PatientHistoryRepository
from src.vector_store.patient_summaries import PatientSummaryStore


def _text(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def _patient_id(name, phone, email):
    slug = re.sub(r'[^a-z0-9]+', '-', (name or '').lower()).strip('-')
    identity = re.sub(r'[^a-z0-9]+', '', f'{phone or email or ""}'.lower())
    return f'dataset-{slug or "patient"}-{identity or uuid.uuid4().hex[:12]}'


def _split_name(name):
    parts = (name or '').split(maxsplit=1)
    return parts[0] if parts else None, parts[1] if len(parts) > 1 else None


def _duplicate_key(name):
    return re.sub(r'\s+', ' ', (name or '').strip()).casefold()


def import_records(workbook_path, database_path, index_vectors=True):
    frame = pd.read_excel(workbook_path, sheet_name='Sheet1')
    required = {'Phone_number', 'Email', 'Name', 'Age', 'Gender', 'Address', 'Summary'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'Missing required workbook columns: {sorted(missing)}')

    store = SQLiteStore(database_path)
    history = PatientHistoryRepository(database_path)
    summary_store = PatientSummaryStore(history) if index_vectors else None
    imported = []
    seen_names = set()
    for _, row in frame.iterrows():
        name = _text(row['Name'])
        if not name:
            raise ValueError('Every imported row must have a Name.')
        duplicate_key = _duplicate_key(name)
        if duplicate_key in seen_names:
            continue
        seen_names.add(duplicate_key)
        first_name, last_name = _split_name(name)
        phone = _text(row['Phone_number'])
        email = _text(row['Email'])
        patient_id = _patient_id(name, phone, email)
        patient = store.save_patient(PatientVO(
            patient_id=patient_id,
            first_name=first_name,
            last_name=last_name,
            gender=_text(row['Gender']),
            age=int(row['Age']) if not pd.isna(row['Age']) else None,
            address=_text(row['Address']),
            email=email,
            mobile_number=phone,
        ))
        summary = _text(row['Summary'])
        if summary:
            history_id = f'{patient_id}-dataset-summary'
            with store._connect() as connection:
                connection.execute(
                    """INSERT INTO medical_history
                    (history_id, patient_id, doctor_id, condition_name, diagnosis_date,
                     treatment, notes, record_type, version, updated_at)
                    VALUES (?, ?, NULL, ?, NULL, NULL, ?, 'note', 1, CURRENT_TIMESTAMP)
                    ON CONFLICT(history_id) DO UPDATE SET
                        condition_name=excluded.condition_name, notes=excluded.notes,
                        updated_at=CURRENT_TIMESTAMP, version=version + 1""",
                    (history_id, patient_id, 'Imported patient summary', summary),
                )
            if summary_store:
                bundle = history.retrieve(requester_patient_id=patient_id, patient_id=patient_id)
                summary_store.index(requester_patient_id=patient_id, bundle=bundle, summary=summary)
        imported.append(patient_id)
    return imported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook', type=Path,
                        default=Path('reference/dataset/records.xlsx'))
    parser.add_argument('--database', default=SQLITE_DB_PATH)
    parser.add_argument('--skip-vectors', action='store_true',
                        help='Import structured data without calling the embedding API.')
    args = parser.parse_args()
    try:
        ids = import_records(args.workbook, args.database, index_vectors=not args.skip_vectors)
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.error(str(error))
    print(f'Imported {len(ids)} patients into {args.database}.')
    if args.skip_vectors:
        print('Summary vectors skipped; rerun without --skip-vectors after configuring OPENAI_API_KEY.')


if __name__ == '__main__':
    main()