"""Seed the SQLite database with sample Indian doctors."""

import argparse

from src.config import SQLITE_DB_PATH
from src.database.sqlite_store import SQLiteStore
from src.models.doctor_vo import DoctorVO


DOCTOR_DATA = [
    ("Amit", "Sharma", "Male", "General Physician", "Bengaluru", "KA"),
    ("Priya", "Rao", "Female", "General Physician", "Bengaluru", "KA"),
    ("Suresh", "Patil", "Male", "General Physician", "Mysuru", "KA"),
    ("Kavita", "Nair", "Female", "General Physician", "Mangaluru", "KA"),
    ("Rahul", "Verma", "Male", "General Physician", "Hubballi", "KA"),
    ("Sneha", "Iyer", "Female", "Pediatrician", "Bengaluru", "KA"),
    ("Vikram", "Mehta", "Male", "Pediatrician", "Chennai", "TN"),
    ("Anjali", "Desai", "Female", "Pediatrician", "Hyderabad", "TG"),
    ("Rakesh", "Kulkarni", "Male", "Orthopedist", "Bengaluru", "KA"),
    ("Meera", "Joshi", "Female", "Orthopedist", "Pune", "MH"),
    ("Nikhil", "Bhat", "Male", "Dentist", "Bengaluru", "KA"),
    ("Shalini", "Menon", "Female", "Dentist", "Kochi", "KL"),
    ("Arjun", "Kapoor", "Male", "Dentist", "New Delhi", "DL"),
    ("Deepak", "Reddy", "Male", "Cardiologist", "Bengaluru", "KA"),
    ("Pooja", "Agarwal", "Female", "Dermatologist", "Jaipur", "RJ"),
    ("Harish", "Gowda", "Male", "Gynecologist", "Bengaluru", "KA"),
    ("Neha", "Srinivasan", "Female", "Obstetrician", "Chennai", "TN"),
    ("Manoj", "Choudhary", "Male", "Neurologist", "New Delhi", "DL"),
    ("Ritu", "Malhotra", "Female", "Psychiatrist", "Gurugram", "HR"),
    ("Sanjay", "Mishra", "Male", "Ophthalmologist", "Lucknow", "UP"),
    ("Divya", "Krishnan", "Female", "ENT Specialist", "Bengaluru", "KA"),
    ("Ashok", "Pillai", "Male", "Gastroenterologist", "Thiruvananthapuram", "KL"),
    ("Farah", "Qureshi", "Female", "Nephrologist", "Mumbai", "MH"),
    ("Karan", "Singh", "Male", "Urologist", "Chandigarh", "CH"),
    ("Swati", "Bansal", "Female", "Endocrinologist", "Ahmedabad", "GJ"),
    ("Yogesh", "Tiwari", "Male", "Pulmonologist", "Bhopal", "MP"),
    ("Lakshmi", "Subramanian", "Female", "Oncologist", "Hyderabad", "TG"),
    ("Vivek", "Chatterjee", "Male", "Radiologist", "Kolkata", "WB"),
    ("Madhuri", "Deshmukh", "Female", "Anesthesiologist", "Nagpur", "MH"),
    ("Sameer", "Khan", "Male", "Physiotherapist", "Bengaluru", "KA"),
]


def build_doctors() -> list[DoctorVO]:
    """Build doctor value objects with Indian contact and license details."""
    doctors = []
    for index, (first_name, last_name, gender, speciality, city, state) in enumerate(
        DOCTOR_DATA, start=1
    ):
        doctors.append(
            DoctorVO(
                doctor_id=f"doctor-{index:03d}",
                first_name=first_name,
                last_name=last_name,
                gender=gender,
                date_of_birth=f"{1965 + index % 25:04d}-{(index % 12) + 1:02d}-{(index % 27) + 1:02d}",
                address=f"{index * 7} Healthcare Avenue, {city}, {state}, India",
                mobile_number=f"+91-98765-{42000 + index:05d}",
                home_number=f"+91-80000-{52000 + index:05d}",
                speciality=speciality,
                license_number=f"IN-MCI-{index:04d}",
            )
        )
    return doctors


def seed_doctors(database_path: str = SQLITE_DB_PATH) -> int:
    """Insert or update the sample doctors and return the record count."""
    database = SQLiteStore(database_path)
    for doctor in build_doctors():
        database.save_doctor(doctor)
    return len(DOCTOR_DATA)


def main() -> None:
    """Parse command-line options and seed the configured database."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        default=SQLITE_DB_PATH,
        help="SQLite database path (default: configured healthcare database)",
    )
    args = parser.parse_args()

    count = seed_doctors(args.database)
    print(f"Seeded {count} doctor records into {args.database}")


if __name__ == "__main__":
    main()
