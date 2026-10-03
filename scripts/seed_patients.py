"""Seed the SQLite database with sample patient records."""

import argparse

from src.config import SQLITE_DB_PATH
from src.database.sqlite_store import SQLiteStore
from src.models.patient_vo import PatientVO


PATIENTS = [
    PatientVO(
        patient_id="patient-001",
        first_name="Aarav",
        last_name="Sharma",
        gender="Male",
        date_of_birth="1988-04-12",
        address="14 MG Road, Bengaluru, Karnataka 560001, India",
        mobile_number="+91-98765-41001",
        home_number="+91-80234-52001",
    ),
    PatientVO(
        patient_id="patient-002",
        first_name="Aditi",
        last_name="Khan",
        gender="Female",
        date_of_birth="1976-09-28",
        address="225 Salt Lake Road, Kolkata, West Bengal 700091, India",
        mobile_number="+91-98765-41002",
        home_number="+91-33234-52002",
    ),
    PatientVO(
        patient_id="patient-003",
        first_name="Rohan",
        last_name="Patel",
        gender="Male",
        date_of_birth="1995-01-17",
        address="87 FC Road, Pune, Maharashtra 411004, India",
        mobile_number="+91-98765-41003",
        home_number="+91-20234-52003",
    ),
    PatientVO(
        patient_id="patient-004",
        first_name="Neha",
        last_name="Perera",
        gender="Female",
        date_of_birth="1969-06-03",
        address="410 FC Road, Pune, Maharashtra 411004, India",
        mobile_number="+91-98765-41004",
        home_number="+91-20234-52004",
    ),
    PatientVO(
        patient_id="patient-005",
        first_name="Arjun",
        last_name="Mehta",
        gender="Male",
        date_of_birth="2001-11-22",
        address="63 Park Street, Kolkata, West Bengal 700016, India",
        mobile_number="+91-98765-41005",
        home_number="+91-33234-52005",
    ),
    PatientVO(
        patient_id="patient-006",
        first_name="Simran",
        last_name="Ahmed",
        gender="Female",
        date_of_birth="1983-03-09",
        address="152 Sector 17 Road, Chandigarh 160017, India",
        mobile_number="+91-98765-41006",
        home_number="+91-17234-52006",
    ),
    PatientVO(
        patient_id="patient-007",
        first_name="Vikram",
        last_name="Gurung",
        gender="Male",
        date_of_birth="1958-12-14",
        address="39 MI Road, Jaipur, Rajasthan 302001, India",
        mobile_number="+91-98765-41007",
        home_number="+91-14134-52007",
    ),
    PatientVO(
        patient_id="patient-008",
        first_name="Kavya",
        last_name="Iyer",
        gender="Female",
        date_of_birth="1992-07-25",
        address="276 Anna Salai, Chennai, Tamil Nadu 600002, India",
        mobile_number="+91-98765-41008",
        home_number="+91-44234-52008",
    ),
    PatientVO(
        patient_id="patient-009",
        first_name="Harsh",
        last_name="Malik",
        gender="Male",
        date_of_birth="1971-10-06",
        address="508 Hazratganj Road, Lucknow, Uttar Pradesh 226001, India",
        mobile_number="+91-98765-41009",
        home_number="+91-52234-52009",
    ),
    PatientVO(
        patient_id="patient-010",
        first_name="Ananya",
        last_name="Reddy",
        gender="Female",
        date_of_birth="2005-02-19",
        address="91 Banjara Hills Road, Hyderabad, Telangana 500034, India",
        mobile_number="+91-98765-41010",
        home_number="+91-40234-52010",
    ),
    PatientVO(
        patient_id="patient-011",
        first_name="Vihaan",
        last_name="Nair",
        gender="Male",
        date_of_birth="2022-03-15",
        address="18 Vyttila Junction, Kochi, Kerala 682019, India",
        mobile_number="+91-98765-41011",
        home_number="+91-48434-52011",
    ),
    PatientVO(
        patient_id="patient-012",
        first_name="Ira",
        last_name="Deshmukh",
        gender="Female",
        date_of_birth="2023-07-08",
        address="42 Civil Lines, Nagpur, Maharashtra 440001, India",
        mobile_number="+91-98765-41012",
        home_number="+91-71234-52012",
    ),
    PatientVO(
        patient_id="patient-013",
        first_name="Kabir",
        last_name="Bose",
        gender="Male",
        date_of_birth="2024-01-21",
        address="76 Alkapuri Road, Vadodara, Gujarat 390007, India",
        mobile_number="+91-98765-41013",
        home_number="+91-26534-52013",
    ),
    PatientVO(
        patient_id="patient-014",
        first_name="Myra",
        last_name="Chatterjee",
        gender="Female",
        date_of_birth="2022-11-30",
        address="29 Boring Road, Patna, Bihar 800001, India",
        mobile_number="+91-98765-41014",
        home_number="+91-61234-52014",
    ),
    PatientVO(
        patient_id="patient-015",
        first_name="Ayaan",
        last_name="Kulkarni",
        gender="Male",
        date_of_birth="2025-05-04",
        address="11 Koregaon Park, Pune, Maharashtra 411001, India",
        mobile_number="+91-98765-41015",
        home_number="+91-20234-52015",
    ),
]


def seed_patients(database_path: str = SQLITE_DB_PATH) -> int:
    """Insert or update the 10 sample patients and return the record count."""
    database = SQLiteStore(database_path)
    for patient in PATIENTS:
        database.save_patient(patient)
    return len(PATIENTS)


def main() -> None:
    """Parse command-line options and seed the configured database."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        default=SQLITE_DB_PATH,
        help="SQLite database path (default: configured healthcare database)",
    )
    args = parser.parse_args()

    count = seed_patients(args.database)
    print(f"Seeded {count} patient records into {args.database}")


if __name__ == "__main__":
    main()
