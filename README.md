# Aviation-Insight: Pilot-Centric Weather Intelligence Platform

Aviation-Insight is a real-time, pilot-centric weather intelligence platform that combines a unified multi-source API hub with a rich interactive dashboard. It decodes and normalizes METAR, TAF, SIGMET, and PIREP data; overlays it on a geospatial map; and surfaces contextual, hover-ready explanations—empowering pilots with mission-ready insights for flight planning and safety.

---

## 🚀 Key Features

### 1. Interactive Flight Route Visualization
- **Flight-Plan Input**: Enter sequences of ICAO waypoints with altitudes (e.g. `KPHX,1500,KBXK,12000,KLAX,50`).
- **Geospatial Map**: Google Maps API–powered US map with plotted waypoints.
- **Altitude Profile**: Dynamic chart of flight-level vs. distance.
- **Hover UI System**: Hover any waypoint or weather overlay to reveal decoded, context-specific definitions (METAR components, SIGMET warnings, etc.).

### 2. Comprehensive Weather Integration
- **METAR** (Current Observations)
- **TAF** (Terminal Aerodrome Forecasts)
- **PIREP** (Pilot Reports)
- **SIGMET/AIRMET** (Hazardous-weather alerts)
- Live fetch from **Aviation Weather Center API**, **avwx.rest**, and **api.weather.gov**.

### 3. Advanced Semantic Parsing
- **Precise Context Tokenization Engine**: Custom contextual tokenizer + text normalization for regex-level parsing of report strings.
- **RAG-Model–Backed Summaries**: Retrieval-augmented generation (RAG) enriches raw data with human-readable interpretations.

### 4. Unified API Hub
- **Multi-Source Endpoints**: Single FastAPI façade for METAR, TAF, PIREP, SIGMET/AIRMET across providers.
- **Consistent Schemas**: Normalized JSON responses, robust error handling, rate-limit resilience.
- **Flexible Querying**: By ICAO, geographic coordinates, or bounding box.

---

## 📦 Technology Stack

| Layer           | Technologies                                                                                  |
| --------------- | --------------------------------------------------------------------------------------------- |
| **Frontend**    | Svelte, Tailwind CSS, Hover UI System, Google Maps API                                         |
| **Backend**     | FastAPI, Python, Avwx.rest, api.weather.gov                                                   |
| **NLP & AI**    | Custom Contextual Tokenizer, Text Normalization, RAG Models                                   |
| **Data Formats**| JSON                                                                                          |
| **DevOps**      | Docker, GitHub Actions (CI/CD), Uvicorn                                                       |

---

## 🔧 Installation & Quick Start

### Backend Setup

1. **Clone repository**
   ```bash
   git clone https://github.com/yourorg/aviation-insight.git
   cd aviation-insight
   ```
2. **Environment**
   Create a `.env` (see `.env.example`):
   ```
   AWC_API_KEY="…"
   AVWX_API_KEY="…"
   CHECKWX_API_KEY="…"
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run API server**
   ```bash
   uvicorn backend.master_run:app --reload --port 8000
   ```

### Frontend Setup

1. **Navigate to frontend**
   ```bash
   cd frontend
   ```
2. **Install dependencies**
   ```bash
   npm install
   ```
3. **Launch Dashboard**
   ```bash
   npm run dev
   ```

---

## 🏗 Architecture Overview

```
/
├─ backend/               # FastAPI application
│  ├─ app/
│  │  ├─ api/             # Endpoints (metar, taf, pirep, sigmet)
│  │  ├─ core/            # Config, rate-limiting, error handling
│  │  ├─ nlp/             # Tokenizer, normalization, RAG integration
│  │  └─ services/        # Multi-provider fetchers (AWC, AVWX, CHECKWX)
│  └─ master_run.py       # Server runner
├─ frontend/              # Svelte + Tailwind dashboard
│  ├─ src/
│  │  ├─ components/      # Interactive map, charts, hover UI
│  │  ├─ stores/          # Svelte stores for live data
│  │  └─ App.svelte       # Main entry
│  └─ tailwind.config.js
├─ .env.example
├─ requirements.txt
└─ README.md
```

---

## 📄 API Reference

Once the API server is running, explore Swagger UI at:

```
http://localhost:8000/docs
```

Core endpoints include:
- **GET /** – Welcome
- **GET /api/v1/health** – Health check
- **GET /api/v1/metar/{icao}** – METAR by ICAO
- **GET /api/v1/taf/{icao}** – TAF by ICAO
- **GET /api/v1/pirep** – PIREP search
- **GET /api/v1/sigmet** – SIGMET/AIRMET search

All endpoints support query by coordinates or bounding box.

---

## 🎯 Usage

### Flight Plan Mode
1. In dashboard, select **Flight Plan**.
2. Paste flight plan string: `ICAO,Altitude,…`.
3. Submit to see route, weather overlays, and alt-profile.

### Station Search Mode
1. Select **Station Search**.
2. Search by ICAO, name, or lat/lon + radius.
3. View detailed METAR/TAF/SIGMET/PIREP.

---

## 🔮 Future Enhancements

- Trend-analysis & short-term forecasting
- Integration with third-party flight-planning tools
- Mobile-optimized PWA
- Offline caching & alerting
- User accounts & saved favorites

---

## 👥 Contributors

- **Chirag** – [Full Stack Web Developer] Lead backend, API integration and Lead Frontend Svelte UI/UX, Tailwnd Designer
- **Smruthi** – Lead Machine Learning, Data Analytics, Researching Tailwind design, data visualization
- **Aditya** – Data processing, NLP engine, documentation, RAG and Artificial Intelligence

