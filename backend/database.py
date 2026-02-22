"""
CivicLens - Database (SQLite)
Simple persistent storage for complaints.
"""

import sqlite3
from typing import Optional

DB_PATH = "civiclens.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id          TEXT PRIMARY KEY,
            filename    TEXT,
            image_url   TEXT,
            issue_type  TEXT,
            severity    TEXT,
            description TEXT,
            department  TEXT,
            lat         REAL DEFAULT 0,
            lng         REAL DEFAULT 0,
            status      TEXT DEFAULT 'Open',
            created_at  TEXT
        )
    """)
    conn.commit()
    conn.close()
    print("[OK] Database initialized")


def save_complaint(record: dict):
    conn = get_conn()
    conn.execute("""
        INSERT INTO complaints
        (id, filename, image_url, issue_type, severity, description, department, lat, lng, status, created_at)
        VALUES (:id, :filename, :image_url, :issue_type, :severity, :description, :department, :lat, :lng, :status, :created_at)
    """, record)
    conn.commit()
    conn.close()


def get_all_complaints(status: Optional[str] = None, issue_type: Optional[str] = None):
    conn = get_conn()
    query = "SELECT * FROM complaints WHERE 1=1"
    params = []
    if status:
        query += " AND status = ?"
        params.append(status)
    if issue_type:
        query += " AND issue_type = ?"
        params.append(issue_type)
    query += " ORDER BY created_at DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_complaint_by_id(complaint_id: str):
    conn = get_conn()
    row = conn.execute("SELECT * FROM complaints WHERE id = ?", (complaint_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_status(complaint_id: str, new_status: str):
    conn = get_conn()
    conn.execute("UPDATE complaints SET status = ? WHERE id = ?", (new_status, complaint_id))
    conn.commit()
    conn.close()