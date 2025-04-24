import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime

from app.api.api import app
from app.schemas.weather import SigmetResponse, AirmetResponse

client = TestClient(app)

@pytest.fixture
def mock_sigmet_data():
    return [
        SigmetResponse(
            source="AWC",
            id="SIGE01",
            raw_text="SIGMET ECHO 1 VALID 250845/251245 KKCI- KZMP MINNEAPOLIS FIR SEV ICE FCST WI 42N 89W - 45N 85W - 47N 91W - 44N 94W - 42N 89W SFC/FL200 MOV NE 10KT WKN",
            phenomenon="ICE",
            valid_from=datetime(2023, 4, 25, 8, 45),
            valid_to=datetime(2023, 4, 25, 12, 45),
            area=[
                {"lat": 42.0, "lon": -89.0},
                {"lat": 45.0, "lon": -85.0},
                {"lat": 47.0, "lon": -91.0},
                {"lat": 44.0, "lon": -94.0},
                {"lat": 42.0, "lon": -89.0}
            ],
            altitude={"lower": 0, "upper": 20000},
            raw_data={}
        )
    ]

@pytest.fixture
def mock_airmet_data():
    return [
        AirmetResponse(
            source="AWC",
            id="CHIT01",
            raw_text="AIRMET TANGO UPDT 1 FOR TURB VALID UNTIL 252000",
            phenomenon="TURB",
            valid_from=datetime(2023, 4, 25, 14, 0),
            valid_to=datetime(2023, 4, 25, 20, 0),
            area=[
                {"lat": 36.0, "lon": -119.0},
                {"lat": 39.0, "lon": -115.0},
                {"lat": 42.0, "lon": -117.0},
                {"lat": 40.0, "lon": -121.0},
                {"lat": 36.0, "lon": -119.0}
            ],
            altitude={"lower": 12000, "upper": 35000},
            raw_data={}
        )
    ]

@patch("app.services.sigmet_service.AWCSigmetService.get_sigmets")
async def test_get_sigmets_awc(mock_get_sigmets, mock_sigmet_data):
    mock_get_sigmets.return_value = mock_sigmet_data
    
    response = client.get("/api/v1/sigmet?source=awc")
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["source"] == "AWC"
    assert data[0]["id"] == "SIGE01"
    assert "raw_text" in data[0]

@patch("app.services.sigmet_service.AVWXSigmetService.get_sigmets")
async def test_get_sigmets_avwx(mock_get_sigmets):
    mock_get_sigmets.return_value = [
        SigmetResponse(
            source="AVWX",
            id="SIGE01",
            raw_text="SIGMET ECHO 1 VALID 250845/251245 KKCI- KZMP MINNEAPOLIS FIR SEV ICE...",
            phenomenon="ICE",
            valid_from=datetime(2023, 4, 25, 8, 45),
            valid_to=datetime(2023, 4, 25, 12, 45),
            raw_data={}
        )
    ]
    
    response = client.get("/api/v1/sigmet?source=avwx")
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["source"] == "AVWX"

@patch("app.services.sigmet_service.AWCAirmetService.get_airmets")
async def test_get_airmets_awc(mock_get_airmets, mock_airmet_data):
    mock_get_airmets.return_value = mock_airmet_data
    
    response = client.get("/api/v1/airmet?source=awc")
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["source"] == "AWC"
    assert data[0]["id"] == "CHIT01"
    assert data[0]["phenomenon"] == "TURB"
    assert "raw_text" in data[0]

@patch("app.services.sigmet_service.AWCSigmetService.get_sigmets")
@patch("app.services.sigmet_service.AVWXSigmetService.get_sigmets")
async def test_get_sigmets_multi(mock_avwx_get_sigmets, mock_awc_get_sigmets, mock_sigmet_data):
    mock_awc_get_sigmets.return_value = mock_sigmet_data
    mock_avwx_get_sigmets.return_value = [
        SigmetResponse(
            source="AVWX",
            id="SIGE02",
            raw_text="SIGMET ECHO 2 VALID 250900/251300...",
            phenomenon="TURB",
            valid_from=datetime(2023, 4, 25, 9, 0),
            valid_to=datetime(2023, 4, 25, 13, 0),
            raw_data={}
        )
    ]
    
    response = client.get("/api/v1/sigmet/multi")
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    # Check that we have one result from each source
    sources = [item["source"] for item in data]
    assert "AWC" in sources
    assert "AVWX" in sources
