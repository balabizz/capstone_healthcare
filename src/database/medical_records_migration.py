"""Idempotent migration for attendant accounts and versioned clinical records."""


def migrate(connection):
    connection.execute('BEGIN IMMEDIATE')
    connection.execute('''CREATE TABLE IF NOT EXISTS attendants (
        attendant_id TEXT PRIMARY KEY, first_name TEXT NOT NULL, last_name TEXT NOT NULL
    )''')
    columns = {r[1] for r in connection.execute('PRAGMA table_info(login_details)')}
    if 'attendant_id' not in columns:
        connection.execute('''CREATE TABLE login_details_new (
            login_id TEXT PRIMARY KEY,
            user_type TEXT NOT NULL CHECK(user_type IN ('patient','doctor','attendant')),
            patient_id TEXT REFERENCES patients(patient_id) ON UPDATE CASCADE ON DELETE CASCADE,
            doctor_id TEXT REFERENCES doctors(doctor_id) ON UPDATE CASCADE ON DELETE CASCADE,
            attendant_id TEXT REFERENCES attendants(attendant_id) ON UPDATE CASCADE ON DELETE CASCADE,
            username TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1 CHECK(is_active IN (0,1)),
            last_login_at TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CHECK ((user_type='patient' AND patient_id IS NOT NULL AND doctor_id IS NULL AND attendant_id IS NULL)
                OR (user_type='doctor' AND doctor_id IS NOT NULL AND patient_id IS NULL AND attendant_id IS NULL)
                OR (user_type='attendant' AND attendant_id IS NOT NULL AND patient_id IS NULL AND doctor_id IS NULL))
        )''')
        connection.execute('''INSERT INTO login_details_new
            (login_id,user_type,patient_id,doctor_id,username,password_hash,is_active,last_login_at,created_at,updated_at)
            SELECT login_id,user_type,patient_id,doctor_id,username,password_hash,is_active,last_login_at,
                   COALESCE(created_at,CURRENT_TIMESTAMP),COALESCE(updated_at,CURRENT_TIMESTAMP)
            FROM login_details''')
        connection.execute('DROP TABLE login_details')
        connection.execute('ALTER TABLE login_details_new RENAME TO login_details')
    for kind in ('patient', 'doctor', 'attendant'):
        connection.execute(f'CREATE INDEX IF NOT EXISTS idx_login_details_{kind} ON login_details({kind}_id)')
    connection.execute('''CREATE TABLE IF NOT EXISTS attendant_patients (
        attendant_id TEXT NOT NULL REFERENCES attendants(attendant_id) ON DELETE CASCADE,
        patient_id TEXT NOT NULL REFERENCES patients(patient_id) ON DELETE CASCADE,
        PRIMARY KEY(attendant_id,patient_id)
    )''')
    columns = {r[1] for r in connection.execute('PRAGMA table_info(medical_history)')}
    for name, definition in {
        'record_type': "TEXT NOT NULL DEFAULT 'diagnosis' CHECK(record_type IN ('diagnosis','note'))",
        'version': 'INTEGER NOT NULL DEFAULT 1',
        'updated_at': 'TEXT',
        'updated_by': 'TEXT REFERENCES attendants(attendant_id)',
    }.items():
        if name not in columns:
            connection.execute(f'ALTER TABLE medical_history ADD COLUMN {name} {definition}')
    connection.execute('''CREATE TABLE IF NOT EXISTS medical_record_revisions (
        revision_id TEXT PRIMARY KEY,
        history_id TEXT NOT NULL REFERENCES medical_history(history_id),
        attendant_id TEXT NOT NULL REFERENCES attendants(attendant_id),
        request_id TEXT NOT NULL,
        request_payload TEXT NOT NULL,
        version INTEGER NOT NULL,
        before_json TEXT,
        after_json TEXT NOT NULL,
        change_reason TEXT NOT NULL,
        changed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(attendant_id,request_id), UNIQUE(history_id,version)
    )''')
    connection.execute('''CREATE TABLE IF NOT EXISTS patient_alerts (
        alert_id TEXT PRIMARY KEY,
        patient_id TEXT NOT NULL REFERENCES patients(patient_id),
        alert_type TEXT NOT NULL CHECK(alert_type IN ('allergy','clinical','other')),
        severity TEXT NOT NULL CHECK(severity IN ('low','moderate','high','critical')),
        description TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','resolved')),
        recorded_by TEXT NOT NULL REFERENCES attendants(attendant_id),
        recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        resolved_by TEXT REFERENCES attendants(attendant_id),
        resolved_at TEXT,
        resolution_reason TEXT
    )''')
    connection.execute('CREATE INDEX IF NOT EXISTS idx_patient_alerts ON patient_alerts(patient_id,status)')
