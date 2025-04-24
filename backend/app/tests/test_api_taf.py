import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime

from app.api.api import app
from app.schemas.weather import TafResponse

client = TestClient(app)

@pytest.mark.asyncio
@patch('app.services.taf_service.AWCTafService.get_taf')
async def test_get_taf_awc(mock_get_taf):
    # Setup mock
    mock_taf = TafResponse(
        source="AWC",
        station="KPHX",
        raw_text="TAF KPHX 201738Z 2018/2118 27015G25KT P6SM FEW120 SCT250 FM210000 25010KT P6SM SCT120 FM211500 23012G22KT P6SM FEW050 BKN120",
        issue_time=datetime.now(),
        valid_from=datetime.now(),
        valid_to=datetime.now(),
        forecast=[
            {
                "from": "2018/2118",
                "wind": "27015G25KT",
                "visibility": "P6SM",
                "clouds": "FEW120 SCT250"
            }
        ]
    )
    mock_get_taf.return_value = mock_taf
    
    # Test
    response = client.get("/api/v1/taf/KPHX?source=awc")
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["station"] == "KPHX"
    assert data["source"] == "AWC"
    assert "raw_text" in data
    assert "forecast" in data

@pytest.mark.asyncio
@patch('app.services.taf_service.AVWXTafService.get_taf')
async def test_get_taf_avwx(mock_get_taf):
    # Setup mock
    mock_taf = TafResponse(
        source="AVWX",
        station="KPHX",
        raw_text="TAF KPHX 201738Z 2018/2118 27015G25KT P6SM FEW120 SCT250 FM210000 25010KT P6SM SCT120 FM211500 23012G22KT P6SM FEW050 BKN120",
        issue_time=datetime.now(),
        valid_from=datetime.now(),
        valid_to=datetime.now(),
        forecast=[
            {
                "wind": {"direction": 270, "speed": 15, "gust": 25, "unit": "kt"},
                "visibility": {"value": 6, "unit": "sm"},
                "clouds": [
                    {"type": "FEW", "height": 120, "unit": "hft"},
                    {"type": "SCT", "height": 250, "unit": "hft"}
                ]
            }
        ]
    )
    mock_get_taf.return_value = mock_taf
    
    # Test
    response = client.get("/api/v1/taf/KPHX?source=avwx")
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["station"] == "KPHX"
    assert data["source"] == "AVWX"
    assert "raw_text" in data
    assert "forecast" in data

@pytest.mark.asyncio
@patch('app.services.taf_service.AWCTafService.get_taf')
@patch('app.services.taf_service.AVWXTafService.get_taf')
async def test_get_taf_multi(mock_avwx_get_taf, mock_awc_get_taf):
    # Setup mocks
    mock_awc_taf = TafResponse(
        source="AWC",
        station="KPHX",
        raw_text="TAF KPHX 201738Z 2018/2118 27015G25KT P6SM FEW120 SCT250",
        issue_time=datetime.now(),
        valid_from=datetime.now(),
        valid_to=datetime.now(),
        forecast=[{"raw": "27015G25KT P6SM FEW120 SCT250"}]
    )
    mock_avwx_taf = TafResponse(
        source="AVWX",
        station="KPHX",
        raw_text="TAF KPHX 201738Z 2018/2118 27015G25KT P6SM FEW120 SCT250",
        issue_time=datetime.now(),
        valid_from=datetime.now(),
        valid_to=datetime.now(),
        forecast=[{"raw": "27015G25KT P6SM FEW120 SCT250"}]
    )
    mock_awc_get_taf.return_value = mock_awc_taf
    mock_avwx_get_taf.return_value = mock_avwx_taf
    
    # Test
    response = client.get("/api/v1/taf/multi/KPHX")
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["source"] == "AWC"
    assert data[1]["source"] == "AVWX"
    assert data[0]["station"] == "KPHX"
    assert data[1]["station"] == "KPHX"
