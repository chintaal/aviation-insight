import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.api.api import app
from app.schemas.weather import PirepResponse

client = TestClient(app)

@pytest.fixture
def mock_pirep_data():
    return [
        PirepResponse(
            source="AWC",
            location="33.43,-112.02",
            raw_text="PIREP KPHX UA /OV KPHX /TM 1800 /FL080 /TP B738 /TB LGT-MOD /RM SMOOTH",
            aircraft_type="B738",
            altitude=8001,
            report_type="UA",
            turbulence={"severity": "LGT-MOD", "floor": 8001, "ceiling": 8001},
            raw_data={}
        )
    ]

@patch("app.services.pirep_service.AWCPirepService.get_pirep_by_station")
async def test_get_pirep_by_station_awc(mock_get_pirep, mock_pirep_data):
    mock_get_pirep.return_value = mock_pirep_data
    
    response = client.get("/api/v1/pirep/station/KPHX?source=awc")
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["source"] == "AWC"
    assert data[0]["location"] == "33.43,-112.02"
    assert "raw_text" in data[0]

@patch("app.services.pirep_service.AVWXPirepService.get_pirep_by_station")
async def test_get_pirep_by_station_avwx(mock_get_pirep, mock_pirep_data):
    mock_get_pirep.return_value = [
        PirepResponse(
            source="AVWX",
            location="33.43,-112.02",
            raw_text="/UA /OV KPHX /TM 1800 /FL080 /TP B738 /TB LGT-MOD /RM SMOOTH",
            aircraft_type="B738",
            altitude=8001,
            report_type="UA",
            turbulence={"severity": "LGT-MOD"},
            raw_data={}
        )
    ]
    
    response = client.get("/api/v1/pirep/station/KPHX?source=avwx")
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["source"] == "AVWX"

@patch("app.services.pirep_service.AWCPirepService.get_pirep_by_location")
async def test_get_pirep_by_location_awc(mock_get_pirep, mock_pirep_data):
    mock_get_pirep.return_value = mock_pirep_data
    
    response = client.get("/api/v1/pirep/location/33.43/-112.02?source=awc")
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["source"] == "AWC"

@patch("app.services.pirep_service.AWCPirepService.get_pirep_by_station")
@patch("app.services.pirep_service.AVWXPirepService.get_pirep_by_station")
async def test_get_pirep_multi_by_station(mock_avwx_get_pirep, mock_awc_get_pirep, mock_pirep_data):
    mock_awc_get_pirep.return_value = mock_pirep_data
    mock_avwx_get_pirep.return_value = [
        PirepResponse(
            source="AVWX",
            location="33.43,-112.02",
            raw_text="/UA /OV KPHX /TM 1800 /FL080 /TP B738 /TB LGT-MOD /RM SMOOTH",
            aircraft_type="B738",
            altitude=8001,
            report_type="UA",
            raw_data={}
        )
    ]
    
    response = client.get("/api/v1/pirep/multi/station/KPHX")
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    # Check that we have one result from each source
    sources = [item["source"] for item in data]
    assert "AWC" in sources
    assert "AVWX" in sources
