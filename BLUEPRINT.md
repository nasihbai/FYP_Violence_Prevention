# Violence Prevention System — Full Stack Blueprint

A real-time violence detection system combining a Python/Flask ML backend with a Vue 3 frontend dashboard. This guide is intended for developers and community contributors who want to run the full system locally or deploy it.

---

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND (Vue 3)                     │
│              nasihbai/Violence_Prevention                │
│                   localhost:5173 (dev)                  │
│                   localhost:3100 (docker)               │
│                                                         │
│  - Admin/Monitoring Dashboard                           │
│  - Real-time alerts via Socket.IO client                │
│  - Role-based access (superadmin/admin/manager/user)    │
└──────────────────┬──────────────────────────────────────┘
                   │  HTTP + Socket.IO
                   │  (REST API + real-time events)
┌──────────────────▼──────────────────────────────────────┐
│                   BACKEND (Flask)                       │
│             nasihbai/FYP_Violence_Prevention             │
│                    localhost:5000                        │
│                                                         │
│  - YOLO v8 multi-person detection                       │
│  - LSTM violence classification                         │
│  - MediaPipe pose extraction                            │
│  - JWT auth + role-based API access                     │
│  - SQLite (default) / PostgreSQL                        │
│  - MJPEG live video feed                                │
└─────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Requirement | Minimum Version | Notes |
|---|---|---|
| Python | 3.10+ | 3.11 recommended |
| Node.js | 18+ | LTS preferred |
| pnpm | 8+ | `npm install -g pnpm` |
| Git | Any | — |
| Webcam or RTSP stream | — | Or use a local video file |
| GPU (optional) | CUDA 11.8+ | Faster YOLO inference; CPU fallback works |

---

## Repository Structure

This system requires **two repositories** cloned side-by-side:

```
your-workspace/
├── FYP_Violence_Prevention/    ← Python backend (ML + API)
└── Violence_Prevention/        ← Vue 3 frontend dashboard
```

---

## Step 1 — Clone Both Repositories

```bash
# Create a working directory
mkdir violence-system && cd violence-system

# Clone the backend
git clone https://github.com/nasihbai/FYP_Violence_Prevention.git

# Clone the frontend (Violence_Prevention branch)
git clone -b Violence_Prevention https://github.com/nasihbai/Violence_Prevention.git
```

---

## Step 2 — Backend Setup (FYP_Violence_Prevention)

### 2.1 Create a Python virtual environment

```bash
cd FYP_Violence_Prevention

# Create virtual environment
python -m venv venv

# Activate — Linux/macOS
source venv/bin/activate

# Activate — Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate — Windows (Command Prompt)
venv\Scripts\activate.bat
```

### 2.2 Install Python dependencies

```bash
pip install -r requirements.txt
```

> This installs TensorFlow, Flask-SocketIO, YOLOv8 (ultralytics), MediaPipe, and other ML/web dependencies. First install may take several minutes.

### 2.3 Verify ML model files exist

The system requires pre-trained LSTM model files inside the `models/` directory:

```
models/
├── violence_lstm_dataset.h5       ← Primary model (required)
├── violence_lstm_enhanced.h5      ← Enhanced variant (optional)
└── violence_lstm_rwf2000.h5       ← RWF2000 dataset model (optional)
```

> If the `models/` folder is empty or missing, see [Training Your Own Model](#optional-training-your-own-model) below.

### 2.4 Configure the backend (optional)

Open `config/settings.py` to adjust key settings:

```python
# Video source
VIDEO_SOURCE = 0          # 0 = default webcam
                          # "/path/to/video.mp4" for a local file
                          # "rtsp://..." for IP cameras

# Detection sensitivity
VIOLENCE_THRESHOLD = 0.6  # 0.0–1.0 (lower = more sensitive)

# YOLO model size (n=fastest, x=most accurate)
YOLO_MODEL = "yolov8n.pt"

# Alert notifications (optional)
EMAIL_ALERTS = False      # Set True + configure SMTP to enable
```

### 2.5 Start the backend server

```bash
python web/app.py
```

You should see:

```
 * Running on http://0.0.0.0:5000
 * Flask-SocketIO server started
```

The backend exposes:
- `http://localhost:5000` — Web dashboard (simple HTML)
- `http://localhost:5000/video_feed` — Live MJPEG video stream
- `http://localhost:5000/api/stats` — JSON detection statistics
- `http://localhost:5000/api/alerts` — Incident history (JWT required)
- `http://localhost:5000/health` — Health check endpoint

---

## Step 3 — Frontend Setup (Violence_Prevention)

### 3.1 Install frontend dependencies

```bash
cd ../Violence_Prevention

pnpm install
```

### 3.2 Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with the following minimum configuration:

```env
# Point the frontend to the Flask backend
VITE_API_URL=http://localhost:5000

# Database (used by Knex for user management — can match backend or be separate)
VITE_DB_CLIENT=sqlite3

# Feature flags
VITE_FEATURE_ADMIN_DASHBOARD=true
VITE_FEATURE_NOTIFICATIONS=true

# Frontend port (for Docker)
WEB_PORT=3100
```

### 3.3 Run database migrations and seed demo users

```bash
pnpm run migrate:latest
pnpm run seed:run
```

This creates the frontend's local database and inserts demo accounts (see [Default Accounts](#default-accounts) below).

### 3.4 Start the frontend dev server

```bash
pnpm run dev
```

Frontend runs at: `http://localhost:5173`

---

## Step 4 — Open the Dashboard

1. Navigate to `http://localhost:5173` in your browser
2. Log in with a demo account (see below)
3. The **Monitoring** page connects to the backend via Socket.IO and displays live detection results

---

## Default Accounts

| Role | Email | Password |
|---|---|---|
| Super Admin | superadmin@example.com | password |
| Admin | admin@example.com | password |
| Manager | manager@example.com | password |
| User | user@example.com | password |

> Change these credentials immediately before any production deployment.

---

## API Reference

All API endpoints are served by the Flask backend on port 5000.

### Authentication

```http
POST /auth/login
Content-Type: application/json

{ "username": "admin", "password": "yourpassword" }
```

Returns a JWT token. Include it in subsequent requests:

```http
Authorization: Bearer <token>
```

### Key Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/health` | No | Server health check |
| GET | `/video_feed` | No | Live MJPEG stream |
| GET | `/api/stats` | No | Real-time detection stats |
| GET | `/api/alerts` | JWT | Incident history |
| GET | `/api/config` | No | Current detection config |
| POST | `/api/config` | JWT | Update thresholds |
| POST | `/api/start` | JWT (manage) | Start detection |
| POST | `/api/stop` | JWT (manage) | Stop detection |
| POST | `/api/reset` | JWT (manage) | Reset statistics |
| GET | `/api/snapshot` | No | Current frame as JPEG |
| POST | `/auth/login` | No | Login (returns JWT) |
| GET | `/auth/me` | JWT | Current user info |

### Socket.IO Events (Real-time)

The frontend subscribes to these events from the backend:

| Event | Direction | Payload |
|---|---|---|
| `violence_alert` | Server → Client | `{ type, confidence, timestamp, location, severity }` |
| `stats_update` | Server → Client | `{ frames_processed, total_detections, fps, uptime_seconds }` |

---

## Video Source Options

Edit `VIDEO_SOURCE` in `config/settings.py`:

| Source Type | Example Value |
|---|---|
| Webcam (default) | `0` |
| Second webcam | `1` |
| Local video file | `"/home/user/video.mp4"` |
| RTSP IP camera | `"rtsp://admin:pass@192.168.1.100:554/stream"` |
| HTTP stream | `"http://192.168.1.100:8080/video"` |

---

## Detection Classes

The LSTM model classifies each person's action sequence into one of these categories:

| Class | Description |
|---|---|
| `neutral` | Normal, non-violent activity |
| `pushing` | Pushing another person |
| `punching` | Punching motions |
| `kicking` | Kicking motions |
| `fighting` | General fighting/brawling |
| `weapon_threat` | Weapon-threatening gestures |

---

## Optional: Docker Deployment (Frontend)

The frontend includes Docker support for production deployment.

```bash
cd Violence_Prevention

# Full stack (app + MySQL database)
pnpm run docker:full

# App only (connect to external database)
pnpm run docker:app-only
```

Or directly with Docker Compose:

```bash
docker compose up -d
```

Frontend will be available at `http://localhost:3100`.

> The backend does not include a Dockerfile. To containerize it, create a `Dockerfile` in `FYP_Violence_Prevention/` using a `python:3.11-slim` base image.

---

## Optional: Training Your Own Model

If you do not have pre-trained model files, you can generate training data and train from scratch.

### Step 1 — Collect pose data

```bash
python pose_data_generation.py
```

Follow the on-screen prompts to record labeled sequences for each violence class.

### Step 2 — Train the LSTM model

```bash
python train_violence_dataset.py
```

Training parameters are in `config/settings.py`:

```python
LSTM_SEQUENCE_LENGTH = 20   # Frames per sequence
LSTM_UNITS = 64             # LSTM hidden units
BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SPLIT = 0.2
```

Trained model is saved to `models/violence_lstm_dataset.h5`.

---

## Optional: Email & Webhook Alerts

Configure in `config/settings.py`:

```python
# Email (Gmail SMTP example)
EMAIL_ALERTS = True
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "youremail@gmail.com"
SMTP_PASSWORD = "your-app-password"   # Use Gmail App Password
ALERT_RECIPIENTS = ["security@example.com"]

# Webhook (Slack/Discord)
WEBHOOK_ALERTS = True
WEBHOOK_URL = "https://hooks.slack.com/services/..."
```

---

## Troubleshooting

### Backend won't start

- Ensure virtual environment is activated before running `python web/app.py`
- Check `requirements.txt` was installed: `pip install -r requirements.txt`
- If TensorFlow fails on Windows, install the CPU-only build: `pip install tensorflow-cpu`

### No video feed / black screen

- Confirm webcam is connected and not in use by another application
- Try changing `VIDEO_SOURCE = 1` if `0` doesn't work
- Test with a local video file first to rule out camera issues

### Frontend shows "Cannot connect to backend"

- Confirm the Flask server is running on port 5000
- Check `VITE_API_URL=http://localhost:5000` in your `.env`
- Ensure CORS is not blocked (Flask-CORS is pre-configured for development)

### Socket.IO not receiving alerts

- Verify both frontend and backend are running simultaneously
- Open browser DevTools → Network → WS to inspect the Socket.IO handshake
- Confirm the backend SocketIO server started (look for `Flask-SocketIO server started` in terminal)

### Model not loading / KeyError on model file

- Ensure `.h5` model files exist in the `models/` directory
- Check `config/settings.py` for the correct `YOLO_MODEL` and LSTM model path
- YOLO model (`yolov8n.pt`) is auto-downloaded from Ultralytics on first run (requires internet)

---

## Project Structure Reference

```
FYP_Violence_Prevention/          ← Backend
├── web/
│   ├── app.py                    ← Flask entry point (run this)
│   └── auth.py                   ← JWT authentication
├── core/
│   ├── detection_engine.py       ← Main detection pipeline
│   ├── yolo_detector.py          ← Multi-person YOLO detection
│   ├── pose_extractor.py         ← MediaPipe pose landmarks
│   └── lstm_model.py             ← LSTM classification
├── database/
│   ├── models.py                 ← SQLAlchemy ORM (User, Stream, Incident, Alert)
│   └── db.py                     ← DB connection
├── config/
│   └── settings.py               ← All configurable settings
├── models/                       ← Pre-trained .h5 model files
├── alerts/                       ← Auto-saved screenshots/clips
└── logs/                         ← Application logs

Violence_Prevention/              ← Frontend
├── src/
│   ├── main.ts                   ← Vue app entry point
│   ├── pages/admin/monitoring/   ← Live monitoring dashboard
│   ├── pages/admin/alerts/       ← Alerts & incidents page
│   ├── stores/monitoring.ts      ← Socket.IO + state management
│   ├── stores/auth.ts            ← JWT auth store
│   └── router/                   ← Route definitions + auth guards
├── migrations/                   ← Knex DB migrations
├── seeds/                        ← Demo user seeding
└── .env                          ← Local environment config
```

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

1. Fork both repositories
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a pull request against the `Violence_Prevention` branch

---

## License

This project was developed as a Final Year Project (FYP). Please check each repository for its specific license terms before using in production.
