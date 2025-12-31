# Aviation Weather API Hub

A unified API for accessing aviation weather data from multiple providers, including the Aviation Weather Center (AWC) and AVWX services. This API hub allows applications to fetch aviation weather information with consistent formatting while leveraging multiple data sources.

## 🌟 Features

- **Multi-Source Data**: Fetch aviation weather data from multiple providers with a single API call
- **Consistent Response Format**: All data is normalized to consistent response schemas
- **Comprehensive Aviation Weather Data**: Support for multiple types of weather data:
  - **METAR**: Current surface weather observations
  - **TAF**: Terminal Aerodrome Forecasts
  - **PIREP**: Pilot Reports
  - **SIGMET/AIRMET**: Significant Meteorological and Airmen's Meteorological Information
- **Flexible Querying**: Fetch data by airport code, coordinates, or bounding boxes
- **Robust Error Handling**: Graceful handling of provider API failures
- **Health Check & API Catalog**: Built-in health monitoring and API discovery

## 📋 Prerequisites

- Python 3.9+
- FastAPI
- Uvicorn
- Other dependencies listed in `requirements.txt`

## 🚀 Quick Start

### Installation

1. Clone this repository
2. Install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

3. Create a `.env` file based on the example:

```
# API keys for weather services
AWC_API_KEY="your_awc_key_here"
CHECKWX_API_KEY="your_checkwx_key_here" 
AVWX_API_KEY="your_avwx_key_here"
```

### Running the Server

Use the master runner script to start the server with various options:

```bash
# Basic usage - starts server at http://localhost:8001
python master_run.py

# Start with browser opening to the API docs
python master_run.py --open

# Use a custom port
python master_run.py --port 3000

# Use a custom environment file
python master_run.py --env-file .env.production
```

### Running Tests

Use the master test runner to execute tests:

```bash
# Run all tests
python master_tests.py

# Run tests with verbose output
python master_tests.py -v

# Run tests for a specific service
python master_tests.py -s metar

# Generate code coverage report
python master_tests.py -c
```

## 📚 API Endpoints

### Core Endpoints

- **Root**: `GET /` - Welcome message and links
- **API Catalog**: `GET /api/v1/catalog` - List of all available endpoints
- **Health Check**: `GET /api/v1/health` - Server health status

### METAR (Surface Observation) Endpoints

- `GET /api/v1/metar/{station}` - Fetch METAR data for a station
  - Query params: `hours`, `source`
- `GET /api/v1/metar/multi/{station}` - Fetch METAR data from multiple sources

### TAF (Terminal Aerodrome Forecast) Endpoints

- `GET /api/v1/taf/{station}` - Fetch TAF data for a station
  - Query params: `hours`, `source`
- `GET /api/v1/taf/multi/{station}` - Fetch TAF data from multiple sources

### PIREP (Pilot Reports) Endpoints

- `GET /api/v1/pirep/station/{station}` - Fetch PIREPs near a station
  - Query params: `hours`, `source`
- `GET /api/v1/pirep/location/{lat}/{lon}` - Fetch PIREPs near coordinates
  - Query params: `radius`, `hours`, `source`
- `GET /api/v1/pirep/area` - Fetch PIREPs in a bounding box
  - Query params: `bbox`, `hours`, `source`
- `GET /api/v1/pirep/multi/station/{station}` - Fetch PIREPs from multiple sources
- `GET /api/v1/pirep/multi/location/{lat}/{lon}` - Fetch PIREPs from multiple sources by location

### SIGMET/AIRMET (Weather Advisories) Endpoints

- `GET /api/v1/sigmet` - Fetch SIGMET data
  - Query params: `region`, `source`
- `GET /api/v1/airmet` - Fetch AIRMET data
  - Query params: `region`, `source`
- `GET /api/v1/sigmet/multi` - Fetch SIGMET data from multiple sources
- `GET /api/v1/airmet/multi` - Fetch AIRMET data from multiple sources

## 🧩 Architecture

### Project Structure

```
backend/
├── app/
│   ├── api/
│   │   ├── v1/
│   │   │   └── endpoints.py   # API endpoints
│   │   └── api.py            # FastAPI app setup
│   ├── core/
│   │   └── config.py         # Settings & configuration
│   ├── schemas/
│   │   └── weather.py        # Pydantic response models
│   ├── services/
│   │   ├── base_client.py    # Base API client
│   │   ├── metar_service.py  # METAR services
│   │   ├── taf_service.py    # TAF services
│   │   ├── pirep_service.py  # PIREP services
│   │   └── sigmet_service.py # SIGMET/AIRMET services
│   └── tests/                # Test modules
├── master_run.py             # Master server runner
├── master_tests.py           # Master test runner
└── requirements.txt          # Python dependencies
```

### Design Pattern

The API follows a layered architecture pattern:

1. **API Layer** (`endpoints.py`): Handles HTTP requests, validates inputs, and returns responses
2. **Service Layer** (services): Implements business logic and integrates with external APIs
3. **Data Model Layer** (schemas): Defines the response data schemas using Pydantic models

This separation of concerns makes the code more maintainable and testable.

## 📑 Data Provider Information

### Aviation Weather Center (AWC)

The official NOAA Aviation Weather Center API provides access to various aviation weather products. This API doesn't require an API key for most endpoints but has rate limiting.

- Base URL: https://aviationweather.gov/data/api/
- Documentation: https://aviationweather.gov/data/api/

### AVWX

AVWX is a comprehensive aviation weather API that provides parsed and decoded weather information. It requires an API key.

- Base URL: https://avwx.rest/api
- Documentation: https://avwx.rest/documentation
- Sign up for API key: https://account.avwx.rest/

## 🔧 Configuration Options

Configuration options are managed through environment variables that can be set in a `.env` file:

| Variable | Description | Required |
|----------|-------------|----------|
| AWC_API_KEY | API key for Aviation Weather Center | No |
| AVWX_API_KEY | API key for AVWX | Yes (for AVWX endpoints) |
| CHECKWX_API_KEY | API key for CheckWX | No |

## 💡 Advanced Usage

### Data Consistency and Fallbacks

When using the multi-provider endpoints, the API attempts to fetch data from all configured providers and returns all successful responses. This provides redundancy if one provider is unavailable.

### Error Handling

The API handles provider errors gracefully:

- If a provider API is down, the response includes an error message but won't fail the entire request
- Invalid requests return appropriate HTTP status codes with descriptive error messages
- Multiple providers ensure redundancy if one provider fails

## 🔄 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License.

## 👏 Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/) for the amazing API framework
- [AVWX](https://avwx.rest/) for providing comprehensive aviation weather data
- [Aviation Weather Center](https://aviationweather.gov/) for their public weather API
