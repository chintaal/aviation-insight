import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from datetime import datetime

from app.api.api import app
from app.schemas.weather import MetarResponse

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "docs_url" in response.json()

@pytest.mark.asyncio
@patch('app.services.metar_service.AWCMetarService.get_metar')
async def test_get_metar_awc(mock_get_metar):
    # Setup mock
    mock_metar = MetarResponse(
        source="AWC",
        station="KPHX",
        raw_text="KPHX 201751Z 27019G35KT 10SM FEW045 FEW250 30/06 A2992 RMK AO2 SLP094 T03000056",
        flight_category="VFR",
        temperature=30.0,
        dewpoint=6.0,
        wind_speed=19,
        wind_direction=270,
        visibility=10,
        timestamp=datetime.now()
    )
    mock_get_metar.return_value = mock_metar
    
    # Test
    response = client.get("/api/v1/metar/KPHX?source=awc")
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["station"] == "KPHX"
    assert data["source"] == "AWC"
    assert "raw_text" in data
    assert data["flight_category"] == "VFR"
    assert data["temperature"] == 30.0

@pytest.mark.asyncio
@patch('app.services.metar_service.AVWXMetarService.get_metar')
async def test_get_metar_avwx(mock_get_metar):
    # Setup mock
    mock_metar = MetarResponse(
        source="AVWX",
        station="KPHX",
        raw_text="KPHX 201751Z 27019G35KT 10SM FEW045 FEW250 30/06 A2992 RMK AO2 SLP094 T03000056",
        flight_category="VFR",
        temperature=30.0,
        dewpoint=6.0,
        wind_speed=19,
        wind_direction=270,
        visibility=10,
        timestamp=datetime.now()
    )
    mock_get_metar.return_value = mock_metar
    
    # Test
    response = client.get("/api/v1/metar/KPHX?source=avwx")
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert data["station"] == "KPHX"
    assert data["source"] == "AVWX"
    assert "raw_text" in data
    assert data["flight_category"] == "VFR"

@pytest.mark.asyncio
@patch('app.services.metar_service.AWCMetarService.get_metar')
@patch('app.services.metar_service.AVWXMetarService.get_metar')
async def test_get_metar_multi(mock_avwx_get_metar, mock_awc_get_metar):
    # Setup mocks
    mock_awc_metar = MetarResponse(
        source="AWC",
        station="KPHX",
        raw_text="KPHX 201751Z 27019G35KT 10SM FEW045 FEW250 30/06 A2992 RMK AO2 SLP094 T03000056",
        flight_category="VFR",
        temperature=30.0,
        dewpoint=6.0,
        timestamp=datetime.now()
    )
    mock_avwx_metar = MetarResponse(
        source="AVWX",
        station="KPHX",
        raw_text="KPHX 201751Z 27019G35KT 10SM FEW045 FEW250 30/06 A2992 RMK AO2 SLP094 T03000056",
        flight_category="VFR",
        temperature=30.0,
        dewpoint=6.0,
        timestamp=datetime.now()
    )
    mock_awc_get_metar.return_value = mock_awc_metar
    mock_avwx_get_metar.return_value = mock_avwx_metar
    
    # Test
    response = client.get("/api/v1/metar/multi/KPHX")
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["source"] == "AWC"
    assert data[1]["source"] == "AVWX"
    assert data[0]["station"] == "KPHX"
    assert data[1]["station"] == "KPHX"
