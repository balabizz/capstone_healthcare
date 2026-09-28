import json
import unittest
from datetime import date

from src.models.patient_vo import PatientVO


class PatientVOAgeTests(unittest.TestCase):
    def test_default_age_is_serializable_none(self):
        patient = PatientVO()
        self.assertIsNone(patient.age)
        self.assertIsNone(patient.calculate_age())
        self.assertIsNone(json.loads(json.dumps(patient.to_dict()))['age'])

    def test_calculation_keeps_stored_age_separate(self):
        patient = PatientVO(age=36, date_of_birth='1986-05-19')
        today = date.today()
        expected = today.year - 1986 - ((today.month, today.day) < (5, 19))
        self.assertEqual(patient.calculate_age(), expected)
        self.assertEqual(patient.age, 36)
        self.assertEqual(PatientVO.from_dict(patient.to_dict()).age, 36)

    def test_missing_and_invalid_birth_dates(self):
        for dob in (None, '', 'invalid'):
            with self.subTest(dob=dob):
                patient = PatientVO(age=36, date_of_birth=dob)
                self.assertIsNone(patient.calculate_age())
                self.assertEqual(patient.age, 36)
