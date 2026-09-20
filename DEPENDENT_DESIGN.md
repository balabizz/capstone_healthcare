# Patient Dependents Management Design

## Overview

The application needs to track patient dependents to support use cases where:
- A patient inquires about a dependent (parent, spouse, child, sibling)
- A dependent may or may not be registered as a patient in the system
- The relationship context is needed for medical queries and appointment booking
- Access control must respect dependent privacy (e.g., adult child may not access parent's records, but parent may access child's)

## 1. Data Model

### 1.1 Relationship Types Enum

```python
# src/models/dependent_types.py
from enum import Enum

class RelationshipType(str, Enum):
    """Enumeration of dependent relationship types."""
    FATHER = "father"
    MOTHER = "mother"
    SPOUSE = "spouse"
    CHILD = "child"
    SIBLING = "sibling"
    GUARDIAN = "guardian"
    OTHER = "other"
```

### 1.2 DependentVO (Value Object)

```python
# src/models/dependent_vo.py
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any
from src.models.dependent_types import RelationshipType

@dataclass
class DependentVO:
    """Value object for a patient's dependent.
    
    A dependent can be:
    1. Another registered patient (dependent_patient_id is set)
    2. An unregistered person (only basic info stored)
    
    All fields are JSON-serializable for UI/Chroma metadata interchange.
    """
    
    dependent_id: Optional[str] = None
    patient_id: Optional[str] = None  # The primary patient who has this dependent
    
    # Dependent person details
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    gender: Optional[str] = None
    date_of_birth: Optional[str] = None  # ISO date string
    
    # Relationship info
    relationship_type: Optional[str] = None  # e.g., 'father', 'spouse', 'child'
    
    # Link to patient record if dependent is registered
    dependent_patient_id: Optional[str] = None
    
    # Emergency/contact info
    mobile_number: Optional[str] = None
    email: Optional[str] = None
    
    # Metadata
    is_primary_contact: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    @property
    def full_name(self) -> str:
        """Return the dependent's full name."""
        fn = (self.first_name or "").strip()
        ln = (self.last_name or "").strip()
        return f"{fn} {ln}".strip()
    
    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable dict."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DependentVO":
        """Create from dict, handling datetime conversions."""
        data = dict(data)
        for field in ["created_at", "updated_at"]:
            val = data.get(field)
            if isinstance(val, datetime):
                data[field] = val.isoformat()
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
```

### 1.3 Database Schema

Add the following tables to `src/database/sqlite_store.py`:

```sql
-- Table: patients_dependents
-- Represents the relationship between a patient and their dependents
CREATE TABLE IF NOT EXISTS patients_dependents (
    dependent_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    
    -- Dependent person details
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    gender TEXT,
    date_of_birth TEXT,
    
    -- Relationship type: father, mother, spouse, child, sibling, guardian, other
    relationship_type TEXT NOT NULL,
    
    -- If the dependent is also a registered patient
    dependent_patient_id TEXT,
    
    -- Contact information
    mobile_number TEXT,
    email TEXT,
    
    -- Flags
    is_primary_contact INTEGER NOT NULL DEFAULT 0 CHECK (is_primary_contact IN (0, 1)),
    
    -- Metadata
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign keys
    FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (dependent_patient_id) REFERENCES patients (patient_id)
        ON UPDATE CASCADE ON DELETE SET NULL
);

-- Index for fast lookup of a patient's dependents
CREATE INDEX IF NOT EXISTS idx_patients_dependents_patient_id
    ON patients_dependents (patient_id);

-- Index for fast lookup of dependent by patient ID (reverse lookup)
CREATE INDEX IF NOT EXISTS idx_patients_dependents_dependent_patient_id
    ON patients_dependents (dependent_patient_id);

-- Table: dependent_access_control
-- Defines who can access whose dependent records (privacy control)
CREATE TABLE IF NOT EXISTS dependent_access_control (
    access_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    dependent_patient_id TEXT NOT NULL,
    
    -- Access type: 'view_only', 'manage', 'book_appointments', etc.
    access_type TEXT NOT NULL,
    
    -- Who granted access (e.g., the dependent themselves, or parent/guardian)
    authorized_by TEXT NOT NULL,
    
    -- When does access expire (NULL = no expiry)
    expires_at TEXT,
    
    -- Metadata
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (dependent_patient_id) REFERENCES patients (patient_id)
        ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_dependent_access_control_patient_id
    ON dependent_access_control (patient_id);
```

## 2. Repository Layer

### 2.1 DependentRepository

```python
# src/repositories/dependent_repository.py
import sqlite3
from typing import List, Optional
from src.models.dependent_vo import DependentVO

class DependentRepository:
    """Repository for managing patient dependents."""
    
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
    
    def add_dependent(self, dependent_vo: DependentVO) -> str:
        """Add a new dependent for a patient."""
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO patients_dependents (
                dependent_id, patient_id, first_name, last_name, gender,
                date_of_birth, relationship_type, dependent_patient_id,
                mobile_number, email, is_primary_contact
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dependent_vo.dependent_id,
            dependent_vo.patient_id,
            dependent_vo.first_name,
            dependent_vo.last_name,
            dependent_vo.gender,
            dependent_vo.date_of_birth,
            dependent_vo.relationship_type,
            dependent_vo.dependent_patient_id,
            dependent_vo.mobile_number,
            dependent_vo.email,
            1 if dependent_vo.is_primary_contact else 0
        ))
        self.connection.commit()
        return dependent_vo.dependent_id
    
    def get_dependents_by_patient(self, patient_id: str) -> List[DependentVO]:
        """Retrieve all dependents for a patient."""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT * FROM patients_dependents
            WHERE patient_id = ?
            ORDER BY is_primary_contact DESC, created_at ASC
        """, (patient_id,))
        
        rows = cursor.fetchall()
        dependents = []
        for row in rows:
            dependents.append(DependentVO.from_dict(dict(row)))
        return dependents
    
    def get_dependent(self, dependent_id: str) -> Optional[DependentVO]:
        """Retrieve a specific dependent."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM patients_dependents WHERE dependent_id = ?", 
                      (dependent_id,))
        row = cursor.fetchone()
        return DependentVO.from_dict(dict(row)) if row else None
    
    def update_dependent(self, dependent_vo: DependentVO) -> None:
        """Update a dependent's information."""
        cursor = self.connection.cursor()
        cursor.execute("""
            UPDATE patients_dependents SET
                first_name = ?, last_name = ?, gender = ?,
                date_of_birth = ?, relationship_type = ?,
                dependent_patient_id = ?, mobile_number = ?, email = ?,
                is_primary_contact = ?, updated_at = CURRENT_TIMESTAMP
            WHERE dependent_id = ?
        """, (
            dependent_vo.first_name,
            dependent_vo.last_name,
            dependent_vo.gender,
            dependent_vo.date_of_birth,
            dependent_vo.relationship_type,
            dependent_vo.dependent_patient_id,
            dependent_vo.mobile_number,
            dependent_vo.email,
            1 if dependent_vo.is_primary_contact else 0,
            dependent_vo.dependent_id
        ))
        self.connection.commit()
    
    def remove_dependent(self, dependent_id: str) -> None:
        """Remove a dependent (soft delete or hard delete based on policy)."""
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM patients_dependents WHERE dependent_id = ?", 
                      (dependent_id,))
        self.connection.commit()
```

## 3. Access Control Layer

### 3.1 DependentAccessControl

```python
# src/repositories/dependent_access_repository.py
import sqlite3
from typing import List, Optional
from datetime import datetime, timedelta

class DependentAccessRepository:
    """Manage access control for dependent records."""
    
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
    
    def grant_access(self, patient_id: str, dependent_patient_id: str,
                     access_type: str, authorized_by: str, 
                     expires_in_days: Optional[int] = None) -> str:
        """Grant access to a dependent's records."""
        access_id = f"access-{patient_id}-{dependent_patient_id}-{int(datetime.now().timestamp())}"
        expires_at = None
        if expires_in_days:
            expires_at = (datetime.now() + timedelta(days=expires_in_days)).isoformat()
        
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT INTO dependent_access_control (
                access_id, patient_id, dependent_patient_id, access_type,
                authorized_by, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (access_id, patient_id, dependent_patient_id, access_type, 
              authorized_by, expires_at))
        self.connection.commit()
        return access_id
    
    def has_access(self, patient_id: str, dependent_patient_id: str,
                   access_type: str) -> bool:
        """Check if patient has access to dependent's records."""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT 1 FROM dependent_access_control
            WHERE patient_id = ? AND dependent_patient_id = ?
            AND access_type = ?
            AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            LIMIT 1
        """, (patient_id, dependent_patient_id, access_type))
        return cursor.fetchone() is not None
```

## 4. UI/Agent Integration

### 4.1 Planner Goal Types

When a patient asks about a dependent, the planner can generate goals like:

```json
{
  "id": "g1",
  "type": "retrieve_dependent_info",
  "depends_on": [],
  "tool": "dependent_manager.get_dependents",
  "arguments": {"patient_id": "patient-123"},
  "requires_confirmation": false
}
```

### 4.2 Agent Event Tracking

Add to `src/repositories/agent_event_repository.py`:

```python
def log_dependent_query(self, patient_id: str, dependent_id: str, 
                        query_type: str, result: str) -> None:
    """Log queries about dependents for audit trail."""
    # Add event with dependent_id in context
```

## 5. Example Workflows

### Scenario 1: Patient Asks About Father's Appointment
```
Input: "When is my father's next appointment?"
→ Planner identifies: need_dependent_info (father), need_appointment_lookup
→ Agent loads patient's dependents, finds dependent with relationship="father"
→ If dependent_patient_id is set, look up appointments for that patient
→ If access_control allows it, return appointment info
→ Respond with appointment details
```

### Scenario 2: Patient Adds New Dependent
```
Input: "Add my spouse as a dependent"
→ UI prompts for spouse details (name, DOB, contact)
→ Agent creates DependentVO and calls dependent_repository.add_dependent()
→ Optionally link to existing patient if spouse already registered
→ Confirm successful addition
```

### Scenario 3: Parent Inquires About Adult Child's Medical Info
```
Input: "What is my adult child's latest prescription?"
→ Check patient's dependents for relationship="child"
→ Check dependent_access_control: does parent have "view_medical" access?
→ If YES: retrieve and display
→ If NO: deny or prompt for permission from adult child
```

## 6. Data Lineage in Chroma

When storing conversation context in Chroma, include dependent metadata:

```python
metadata = {
    "patient_id": "patient-123",
    "dependent_id": "dependent-456",  # New field
    "dependent_relationship": "father",
    "dependent_name": "John Senior",
    "summary_version": "1.0",
    "source_updated_at": "2024-09-20T...",
    "content_type": "conversation_context"
}
```

## 7. Privacy & Security Considerations

1. **Dependent visibility**: Only show dependents to authorized users
2. **Cross-dependent access**: Adult children don't see parents' medical data by default
3. **Access expiry**: Time-bound access permissions (e.g., during hospital stay)
4. **Audit trail**: Log all dependent record accesses
5. **Data minimization**: Store only necessary dependent info for unregistered dependents

## 8. Implementation Roadmap

1. Create `src/models/dependent_vo.py` with DependentVO dataclass
2. Create `src/models/dependent_types.py` with RelationshipType enum
3. Update `src/database/sqlite_store.py` to add dependent tables
4. Create `src/repositories/dependent_repository.py`
5. Create `src/repositories/dependent_access_repository.py`
6. Add dependent query handlers to the agent executor
7. Update planner to recognize dependent-related goals
8. Add tests in `tests/test_dependent_repository.py`
9. Update Streamlit UI to display and manage dependents
