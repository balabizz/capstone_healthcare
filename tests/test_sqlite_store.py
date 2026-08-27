from src.database.sqlite_store import SQLiteStore
from src.models.doctor_vo import DoctorVO
from src.models.patient_vo import PatientVO


def test_initializes_patient_and_doctor_tables(tmp_path):
    database = SQLiteStore(str(tmp_path / "healthcare.db"))

    with database._connect() as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert {
        "patients",
        "doctors",
        "login_details",
        "appointments",
        "medical_history",
        "prescriptions",
        "billing",
    }.issubset(tables)


def test_related_tables_use_normalized_foreign_keys(tmp_path):
    database = SQLiteStore(str(tmp_path / "healthcare.db"))

    with database._connect() as connection:
        appointment_foreign_keys = connection.execute(
            "PRAGMA foreign_key_list(appointments)"
        ).fetchall()
        prescription_foreign_keys = connection.execute(
            "PRAGMA foreign_key_list(prescriptions)"
        ).fetchall()
        billing_foreign_keys = connection.execute(
            "PRAGMA foreign_key_list(billing)"
        ).fetchall()

    assert {row[2] for row in appointment_foreign_keys} == {"patients", "doctors"}
    assert {row[2] for row in prescription_foreign_keys} == {
        "patients", "doctors", "appointments"
    }
    assert {row[2] for row in billing_foreign_keys} == {"patients", "appointments"}


def test_login_details_store_patient_and_doctor_credentials(tmp_path):
    database = SQLiteStore(str(tmp_path / "healthcare.db"))

    with database._connect() as connection:
        connection.execute(
            "INSERT INTO patients (patient_id) VALUES (?)", ("p-1",)
        )
        connection.execute(
            "INSERT INTO doctors (doctor_id) VALUES (?)", ("d-1",)
        )
        connection.execute(
            """
            INSERT INTO login_details (
                login_id, user_type, patient_id, username, password_hash
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("login-p-1", "patient", "p-1", "patient@example.com", "hashed-value"),
        )
        connection.execute(
            """
            INSERT INTO login_details (
                login_id, user_type, doctor_id, username, password_hash
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("login-d-1", "doctor", "d-1", "doctor@example.com", "hashed-value"),
        )

        rows = connection.execute(
            "SELECT user_type, username, password_hash FROM login_details "
            "ORDER BY login_id"
        ).fetchall()

    assert [tuple(row) for row in rows] == [
        ("doctor", "doctor@example.com", "hashed-value"),
        ("patient", "patient@example.com", "hashed-value"),
    ]


def test_saves_and_loads_patient(tmp_path):
    database = SQLiteStore(str(tmp_path / "healthcare.db"))
    patient = PatientVO(
        patient_id="p-1",
        first_name="John",
        last_name="Doe",
        date_of_birth="1990-01-01",
        address="123 Main St",
        mobile_number="+1000000000",
    )

    database.save_patient(patient)

    stored_patient = database.get_patient("p-1")
    assert stored_patient is not None
    assert stored_patient.patient_id == "p-1"
    assert stored_patient.full_name == "John Doe"
    assert stored_patient.date_of_birth == "1990-01-01"
    assert stored_patient.created_at is not None
    assert stored_patient.updated_at is not None


def test_saves_and_loads_doctor(tmp_path):
    database = SQLiteStore(str(tmp_path / "healthcare.db"))
    doctor = DoctorVO(
        doctor_id="d-1",
        first_name="Jane",
        last_name="Smith",
        speciality="Nephrologist",
        license_number="LIC-001",
    )

    database.save_doctor(doctor)

    stored_doctor = database.get_doctor("d-1")
    assert stored_doctor is not None
    assert stored_doctor.doctor_id == "d-1"
    assert stored_doctor.full_name == "Jane Smith"
    assert stored_doctor.speciality == "Nephrologist"
    assert stored_doctor.license_number == "LIC-001"
