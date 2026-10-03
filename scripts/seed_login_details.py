"""Populate login details for every patient and doctor."""

import argparse
import sqlite3

from src.config import SQLITE_DB_PATH
from src.database.sqlite_store import SQLiteStore


def populate_login_details(database_path: str = SQLITE_DB_PATH) -> int:
    """Create or update login accounts for every patient and doctor."""
    SQLiteStore(database_path)
    connection = sqlite3.connect(database_path)
    try:
        people = connection.execute(
            """
            SELECT 'patient' AS user_type, patient_id AS person_id,
                   first_name, last_name
            FROM patients
            UNION ALL
            SELECT 'doctor', doctor_id, first_name, last_name
            FROM doctors
            """
        ).fetchall()

        usernames = [
            f"{first_name.strip().lower()}.{last_name.strip().lower()}"
            for _, _, first_name, last_name in people
        ]
        if len(usernames) != len(set(usernames)):
            raise ValueError("Patient and doctor names must produce unique usernames")

        for user_type, person_id, first_name, last_name in people:
            username = f"{first_name.strip().lower()}.{last_name.strip().lower()}"
            login_id = f"{user_type}-{person_id}"
            password = "patient" if user_type == "patient" else "doctor"
            patient_id = person_id if user_type == "patient" else None
            doctor_id = person_id if user_type == "doctor" else None

            connection.execute(
                """
                INSERT INTO login_details (
                    login_id, user_type, patient_id, doctor_id,
                    username, password_hash
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(login_id) DO UPDATE SET
                    user_type = excluded.user_type,
                    patient_id = excluded.patient_id,
                    doctor_id = excluded.doctor_id,
                    username = excluded.username,
                    password_hash = excluded.password_hash,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    login_id,
                    user_type,
                    patient_id,
                    doctor_id,
                    username,
                    password,
                ),
            )

        test_accounts = (
            ("test-patient-login", "patient", "patient-001", None, "testpatient", "patient"),
            ("test-doctor-login", "doctor", None, "doctor-001", "testdoctor", "doctor"),
        )
        for login_id, user_type, patient_id, doctor_id, username, password in test_accounts:
            connection.execute(
                """
                INSERT INTO login_details (
                    login_id, user_type, patient_id, doctor_id,
                    username, password_hash
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(login_id) DO UPDATE SET
                    user_type = excluded.user_type,
                    patient_id = excluded.patient_id,
                    doctor_id = excluded.doctor_id,
                    username = excluded.username,
                    password_hash = excluded.password_hash,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (login_id, user_type, patient_id, doctor_id, username, password),
            )

        connection.commit()
        return len(people) + len(test_accounts)
    finally:
        connection.close()


def main() -> None:
    """Parse command-line options and populate the configured database."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        default=SQLITE_DB_PATH,
        help="SQLite database path (default: configured healthcare database)",
    )
    args = parser.parse_args()

    count = populate_login_details(args.database)
    print(f"Populated {count} login records into {args.database}")


if __name__ == "__main__":
    main()
