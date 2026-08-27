"""SQLite persistence for structured patient and doctor records."""

import sqlite3
from pathlib import Path
from typing import Optional

from src.models.doctor_vo import DoctorVO
from src.models.patient_vo import PatientVO


class SQLiteStore:
    """Persist EHR entities in a local SQLite database."""

    def __init__(self, database_path: str):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        """Create the initial EHR schema when it does not exist."""
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY,
                    first_name TEXT,
                    last_name TEXT,
                    gender TEXT,
                    date_of_birth TEXT,
                    address TEXT,
                    mobile_number TEXT,
                    home_number TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS doctors (
                    doctor_id TEXT PRIMARY KEY,
                    first_name TEXT,
                    last_name TEXT,
                    gender TEXT,
                    date_of_birth TEXT,
                    address TEXT,
                    mobile_number TEXT,
                    home_number TEXT,
                    speciality TEXT,
                    license_number TEXT UNIQUE,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS login_details (
                    login_id TEXT PRIMARY KEY,
                    user_type TEXT NOT NULL CHECK (user_type IN ('patient', 'doctor')),
                    patient_id TEXT,
                    doctor_id TEXT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
                    last_login_at TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    CHECK (
                        (user_type = 'patient' AND patient_id IS NOT NULL AND doctor_id IS NULL)
                        OR (user_type = 'doctor' AND doctor_id IS NOT NULL AND patient_id IS NULL)
                    ),
                    FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
                        ON UPDATE CASCADE ON DELETE CASCADE,
                    FOREIGN KEY (doctor_id) REFERENCES doctors (doctor_id)
                        ON UPDATE CASCADE ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_login_details_patient
                    ON login_details (patient_id);
                CREATE INDEX IF NOT EXISTS idx_login_details_doctor
                    ON login_details (doctor_id);

                CREATE TABLE IF NOT EXISTS appointments (
                    appointment_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    doctor_id TEXT NOT NULL,
                    appointment_datetime TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'scheduled',
                    reason TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
                        ON UPDATE CASCADE ON DELETE RESTRICT,
                    FOREIGN KEY (doctor_id) REFERENCES doctors (doctor_id)
                        ON UPDATE CASCADE ON DELETE RESTRICT
                );

                CREATE TABLE IF NOT EXISTS medical_history (
                    history_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    doctor_id TEXT,
                    condition_name TEXT NOT NULL,
                    diagnosis_date TEXT,
                    treatment TEXT,
                    notes TEXT,
                    recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
                        ON UPDATE CASCADE ON DELETE CASCADE,
                    FOREIGN KEY (doctor_id) REFERENCES doctors (doctor_id)
                        ON UPDATE CASCADE ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS prescriptions (
                    prescription_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    doctor_id TEXT NOT NULL,
                    appointment_id TEXT,
                    medication_name TEXT NOT NULL,
                    dosage TEXT,
                    frequency TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    instructions TEXT,
                    prescribed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
                        ON UPDATE CASCADE ON DELETE CASCADE,
                    FOREIGN KEY (doctor_id) REFERENCES doctors (doctor_id)
                        ON UPDATE CASCADE ON DELETE RESTRICT,
                    FOREIGN KEY (appointment_id) REFERENCES appointments (appointment_id)
                        ON UPDATE CASCADE ON DELETE SET NULL
                );

                CREATE TABLE IF NOT EXISTS billing (
                    billing_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    appointment_id TEXT,
                    invoice_number TEXT UNIQUE,
                    amount REAL NOT NULL DEFAULT 0,
                    currency TEXT NOT NULL DEFAULT 'USD',
                    status TEXT NOT NULL DEFAULT 'pending',
                    billed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    paid_at TEXT,
                    FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
                        ON UPDATE CASCADE ON DELETE RESTRICT,
                    FOREIGN KEY (appointment_id) REFERENCES appointments (appointment_id)
                        ON UPDATE CASCADE ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_appointments_patient
                    ON appointments (patient_id);
                CREATE INDEX IF NOT EXISTS idx_appointments_doctor
                    ON appointments (doctor_id);
                CREATE INDEX IF NOT EXISTS idx_medical_history_patient
                    ON medical_history (patient_id);
                CREATE INDEX IF NOT EXISTS idx_prescriptions_patient
                    ON prescriptions (patient_id);
                CREATE INDEX IF NOT EXISTS idx_billing_patient
                    ON billing (patient_id);
                """
            )

    def save_patient(self, patient: PatientVO) -> PatientVO:
        """Insert or update a patient record and return the stored value object."""
        if not patient.patient_id:
            raise ValueError("patient_id is required")
        values = patient.to_dict()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO patients (
                    patient_id, first_name, last_name, gender, date_of_birth,
                    address, mobile_number, home_number, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), CURRENT_TIMESTAMP)
                ON CONFLICT(patient_id) DO UPDATE SET
                    first_name = excluded.first_name,
                    last_name = excluded.last_name,
                    gender = excluded.gender,
                    date_of_birth = excluded.date_of_birth,
                    address = excluded.address,
                    mobile_number = excluded.mobile_number,
                    home_number = excluded.home_number,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    values["patient_id"], values["first_name"], values["last_name"],
                    values["gender"], values["date_of_birth"], values["address"],
                    values["mobile_number"], values["home_number"], values["created_at"],
                ),
            )
        return self.get_patient(patient.patient_id)  # type: ignore[return-value]

    def get_patient(self, patient_id: str) -> Optional[PatientVO]:
        """Retrieve a patient by ID."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM patients WHERE patient_id = ?", (patient_id,)
            ).fetchone()
        return PatientVO.from_dict(dict(row)) if row else None

    def save_doctor(self, doctor: DoctorVO) -> DoctorVO:
        """Insert or update a doctor record and return the stored value object."""
        if not doctor.doctor_id:
            raise ValueError("doctor_id is required")
        values = doctor.to_dict()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO doctors (
                    doctor_id, first_name, last_name, gender, date_of_birth,
                    address, mobile_number, home_number, speciality, license_number,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), CURRENT_TIMESTAMP)
                ON CONFLICT(doctor_id) DO UPDATE SET
                    first_name = excluded.first_name,
                    last_name = excluded.last_name,
                    gender = excluded.gender,
                    date_of_birth = excluded.date_of_birth,
                    address = excluded.address,
                    mobile_number = excluded.mobile_number,
                    home_number = excluded.home_number,
                    speciality = excluded.speciality,
                    license_number = excluded.license_number,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    values["doctor_id"], values["first_name"], values["last_name"],
                    values["gender"], values["date_of_birth"], values["address"],
                    values["mobile_number"], values["home_number"], values["speciality"],
                    values["license_number"], values["created_at"],
                ),
            )
        return self.get_doctor(doctor.doctor_id)  # type: ignore[return-value]

    def get_doctor(self, doctor_id: str) -> Optional[DoctorVO]:
        """Retrieve a doctor by ID."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM doctors WHERE doctor_id = ?", (doctor_id,)
            ).fetchone()
        return DoctorVO.from_dict(dict(row)) if row else None
