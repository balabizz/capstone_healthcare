"""Doctor value object for UI and Chroma DB metadata interchange.

This module defines DoctorVO, a lightweight, JSON-serializable value object
suited for passing doctor data between UI layers and Chroma vector store
metadata. Dates are represented as ISO strings to keep the metadata JSON-safe.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Optional, Dict, Any


@dataclass
class DoctorVO:
    """Value object for doctor data used by UI and Chroma DB metadata.

    All fields are JSON-serializable (dates as ISO strings). Keep this object
    lightweight and free of complex behaviour so it's safe to use as metadata
    stored alongside vectors in Chroma or exchanged with front-end UIs.
    """

    doctor_id: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    gender: Optional[str] = None
    date_of_birth: Optional[str] = None  # ISO date string: YYYY-MM-DD
    address: Optional[str] = None
    mobile_number: Optional[str] = None
    home_number: Optional[str] = None
    speciality: Optional[str] = None
    license_number: Optional[str] = None
    created_at: Optional[str] = None  # ISO datetime
    updated_at: Optional[str] = None  # ISO datetime

    @property
    def full_name(self) -> str:
        """Return the full name constructed from first and last name."""
        fn = (self.first_name or "").strip()
        ln = (self.last_name or "").strip()
        return f"{fn} {ln}".strip()

    def age(self) -> Optional[int]:
        """Return computed age in years if date_of_birth is available, else None.

        Note: date_of_birth is stored as an ISO string in this VO. Parsing is
        forgiving; invalid formats return None.
        """
        dob = self.date_of_birth
        if not dob:
            return None
        try:
            if isinstance(dob, str):
                dt = datetime.fromisoformat(dob).date()
            elif isinstance(dob, date):
                dt = dob
            else:
                return None
        except Exception:
            return None

        today = date.today()
        years = today.year - dt.year
        if (today.month, today.day) < (dt.month, dt.day):
            years -= 1
        return years

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict representation of the DoctorVO."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DoctorVO":
        """Create a DoctorVO from a mapping. Accepts date strings or date objects.

        Any non-serializable values should be converted before calling this
        method.
        """
        data = dict(data)
        dob = data.get("date_of_birth")
        if isinstance(dob, (date, datetime)):
            data["date_of_birth"] = dob.isoformat()
        created = data.get("created_at")
        if isinstance(created, datetime):
            data["created_at"] = created.isoformat()
        updated = data.get("updated_at")
        if isinstance(updated, datetime):
            data["updated_at"] = updated.isoformat()
        return cls(**data)

    def to_metadata(self) -> Dict[str, Any]:
        """Alias for to_dict; kept for semantic clarity when storing in Chroma."""
        return self.to_dict()

    @classmethod
    def from_metadata(cls, meta: Dict[str, Any]) -> "DoctorVO":
        """Create a DoctorVO from Chroma metadata (same shape as to_metadata)."""
        return cls.from_dict(meta)

    def with_updates(self, **kwargs: Any) -> "DoctorVO":
        """Return a shallow copy of the VO with provided fields updated.

        Example: vo = vo.with_updates(address="New Addr", mobile_number="...")
        """
        data = self.to_dict()
        data.update(kwargs)
        return DoctorVO.from_dict(data)

    def __repr__(self) -> str:  # pragma: no cover - readability
        return (
            f"DoctorVO(doctor_id={self.doctor_id!r}, name={self.full_name!r}, "
            f"speciality={self.speciality!r}, dob={self.date_of_birth!r})"
        )
