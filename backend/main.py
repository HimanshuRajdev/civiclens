"""
CivicLens - FastAPI Backend
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import uuid
import os
from datetime import datetime
from pathlib import Path

from inference_efficientnet import run_inference
from complaint import generate_complaint
from database import init_db, save_complaint, get_all_complaints, get_complaint_by_id, update_status
from duplicate_check import check_duplicate

app = FastAPI(title="CivicLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.on_event("startup")
def startup():
    init_db()
    print("[OK] CivicLens backend started")


@app.get("/")
def root():
    return {"status": "CivicLens API running"}


@app.post("/report")
async def report(file: UploadFile = File(...), lat: float = 0.0, lng: float = 0.0):
    """Upload image → AI detection + GPT-4o Vision complaint generation."""

    ext = Path(file.filename).suffix or ".jpg"
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(await file.read())

    # Run model inference
    detection = run_inference(filepath)
    if not detection:
        raise HTTPException(status_code=422, detail="No civic issue detected. Try a clearer photo.")

    # Check for duplicates
    duplicate = check_duplicate(detection["class"], lat, lng)

    # Generate complaint using GPT-4o Vision
    complaint = generate_complaint(detection, image_path=filepath)

    return {
        "image_url": f"/uploads/{filename}",
        "filename": filename,
        "detection": detection,
        "complaint": complaint,
        "location": {"lat": lat, "lng": lng},
        "duplicate": duplicate,
    }


@app.post("/submit")
async def submit(data: dict):
    """Save confirmed complaint to DB."""
    complaint_id = f"CL-{uuid.uuid4().hex[:8].upper()}"
    record = {
        "id": complaint_id,
        "filename": data.get("filename"),
        "image_url": data.get("image_url"),
        "issue_type": data.get("issue_type"),
        "severity": data.get("severity"),
        "description": data.get("description"),
        "department": data.get("department"),
        "lat": data.get("lat", 0.0),
        "lng": data.get("lng", 0.0),
        "status": "Open",
        "created_at": datetime.utcnow().isoformat(),
    }
    save_complaint(record)
    return {"complaint_id": complaint_id, "status": "Open"}


@app.get("/complaints")
def complaints(status: str = None, issue_type: str = None):
    return get_all_complaints(status=status, issue_type=issue_type)


@app.get("/complaints/{complaint_id}")
def complaint_detail(complaint_id: str):
    record = get_complaint_by_id(complaint_id)
    if not record:
        raise HTTPException(status_code=404, detail="Complaint not found")
    return record


@app.patch("/complaints/{complaint_id}/status")
def update_complaint_status(complaint_id: str, data: dict):
    new_status = data.get("status")
    if new_status not in ["Open", "In Progress", "Resolved"]:
        raise HTTPException(status_code=400, detail="Invalid status")
    update_status(complaint_id, new_status)
    return {"complaint_id": complaint_id, "status": new_status}


@app.get("/stats")
def stats():
    all_c = get_all_complaints()
    total = len(all_c)
    by_type = {}
    by_status = {"Open": 0, "In Progress": 0, "Resolved": 0}
    by_severity = {"High": 0, "Medium": 0, "Low": 0}
    for c in all_c:
        t = c.get("issue_type", "Unknown")
        by_type[t] = by_type.get(t, 0) + 1
        by_status[c.get("status", "Open")] = by_status.get(c.get("status", "Open"), 0) + 1
        by_severity[c.get("severity", "Low")] = by_severity.get(c.get("severity", "Low"), 0) + 1
    return {"total": total, "by_type": by_type, "by_status": by_status, "by_severity": by_severity}


@app.get("/stats/timeline")
def stats_timeline():
    """Complaints grouped by date for trend charts."""
    all_c = get_all_complaints()
    from collections import defaultdict
    daily = defaultdict(int)
    for c in all_c:
        date_str = c.get("created_at", "")[:10]
        if date_str:
            daily[date_str] += 1
    timeline = [{"date": k, "count": v} for k, v in sorted(daily.items())]
    return timeline


@app.get("/stats/departments")
def stats_departments():
    """Per-department breakdown of complaints."""
    all_c = get_all_complaints()
    dept_map = {
        "Pothole": "Roads & Infrastructure",
        "Sinkhole": "Roads & Infrastructure",
        "Water Leakage": "Water & Sewage",
        "Garbage Overflow": "Sanitation",
        "Broken Streetlight": "Electrical",
        "Broken Sidewalk": "Public Works",
    }
    from collections import defaultdict
    depts = defaultdict(lambda: {"total": 0, "open": 0, "in_progress": 0, "resolved": 0})
    for c in all_c:
        dept = dept_map.get(c.get("issue_type", ""), c.get("department", "Other"))
        depts[dept]["total"] += 1
        status = c.get("status", "Open")
        if status == "Open":
            depts[dept]["open"] += 1
        elif status == "In Progress":
            depts[dept]["in_progress"] += 1
        elif status == "Resolved":
            depts[dept]["resolved"] += 1
    return [{"department": k, **v} for k, v in depts.items()]


@app.get("/stats/resolution")
def stats_resolution():
    """Resolution metrics."""
    all_c = get_all_complaints()
    total = len(all_c)
    resolved = sum(1 for c in all_c if c.get("status") == "Resolved")
    open_count = sum(1 for c in all_c if c.get("status") == "Open")
    in_progress = sum(1 for c in all_c if c.get("status") == "In Progress")
    rate = round((resolved / total) * 100, 1) if total > 0 else 0
    return {
        "total": total,
        "resolved": resolved,
        "open": open_count,
        "in_progress": in_progress,
        "resolution_rate": rate,
    }


@app.delete("/complaints/{complaint_id}")
def delete_complaint(complaint_id: str):
    """Delete a complaint by ID."""
    from database import get_conn
    conn = get_conn()
    result = conn.execute("DELETE FROM complaints WHERE id = ?", (complaint_id,))
    conn.commit()
    deleted = result.rowcount
    conn.close()
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Complaint not found")
    return {"deleted": complaint_id}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)