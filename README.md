# CivicLens 🏙️
**AI-Powered Civic Issue Reporting** · Built at MadData 2026, UW–Madison

> Snap it. Report it. Fix it.

---

## What It Does

Citizens snap a photo of any road issue. AI classifies it, writes the complaint, routes it to the right city department, and pins it on a live map — in under 10 seconds.

**6 detectable issues:** Potholes · Cracked Pavement · Road Debris · Broken Signs · Faded Lane Markings · Normal Road

**96.8% test accuracy** on 7,200 training images.

---

## Stack

- **ML:** EfficientNetV2-S (PyTorch) — 96.8% accuracy, 3-zone confidence gate
- **LLM:** Google Gemini Vision — complaint generation
- **Backend:** FastAPI + SQLite
- **Mobile:** React Native + Expo
- **Other:** Haversine deduplication, react-native-maps, GPS capture

---

## Setup

### Backend
```bash
cd backend
python3.11 -m venv env && source env/bin/activate
pip install fastapi uvicorn torch==2.5.1 torchvision==0.20.1 pillow python-multipart google-generativeai
# Add best_roadscan.pt to backend/
# Set GEMINI_API_KEY in .env
python main.py
```

### Mobile App
```bash
cd civiclens-mobile
npm install
# Update API_URL in config.js to your machine's local IP
npx expo start
```

> Phone and computer must be on the same WiFi network.

---

## Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/report` | Upload image → detect → generate complaint |
| `POST` | `/submit` | Save complaint to DB |
| `GET` | `/complaints` | List all complaints |
| `PATCH` | `/complaints/{id}/status` | Update status |
| `GET` | `/stats` | Analytics |

---

## Environment Variables

```
GEMINI_API_KEY=your_key_here
```

---

## Team
Built in 24 hours at MadData 2026 · University of Wisconsin–Madison
