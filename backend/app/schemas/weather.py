from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime

class WeatherResponseBase(BaseModel):
    source: str
    timestamp: datetime = Field(default_factory=datetime.now)
    raw_data: Optional[Any] = None

class MetarResponse(WeatherResponseBase):
    station: str
    raw_text: Optional[str] = None
    flight_category: Optional[str] = None
    temperature: Optional[float] = None
    dewpoint: Optional[float] = None
    wind_speed: Optional[float] = None
    wind_direction: Optional[int] = None
    visibility: Optional[float] = None
    ceiling: Optional[int] = None
    clouds: Optional[List[Dict[str, Any]]] = None
    parsed_metar: Optional[Dict[str, Any]] = None  # Detailed parsed METAR data
    pilot_summary: Optional[str] = None  # Pilot-friendly summary of the METAR

class TafResponse(WeatherResponseBase):
    station: str
    raw_text: Optional[str] = None
    forecast: Optional[List[Dict[str, Any]]] = None
    issue_time: Optional[datetime] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    parsed_taf: Optional[Dict[str, Any]] = None  # Detailed parsed TAF data
    pilot_summary: Optional[str] = None  # Pilot-friendly summary of the TAF

class PirepResponse(WeatherResponseBase):
    location: str
    raw_text: Optional[str] = None
    aircraft_type: Optional[str] = None
    altitude: Optional[Union[int, str]] = None
    report_type: Optional[str] = None
    turbulence: Optional[Dict[str, Any]] = None
    icing: Optional[Dict[str, Any]] = None
    sky_conditions: Optional[str] = None
    remarks: Optional[str] = None
    timestamp: Optional[str] = None

class EnhancedPirepResponse(PirepResponse):
    """Enhanced PIREP response with additional fields for cockpit display"""
    severity_level: Optional[str] = None  # Overall severity level (light, moderate, severe)
    distance_from_station: Optional[float] = None  # Distance in nm from reference station
    is_relevant: Optional[bool] = None  # Relevance flag based on altitude, recency
    hazard_summary: Optional[str] = None  # Brief summary of hazards
    position_data: Optional[Dict[str, Any]] = None  # Lat/long and other position info
    parsed_pirep: Optional[Dict[str, Any]] = None  # Structured data from the raw text

class SigmetResponse(WeatherResponseBase):
    id: str
    raw_text: Optional[str] = None
    area: Optional[List[Dict[str, float]]] = None
    altitude: Optional[Dict[str, int]] = None
    phenomenon: Optional[str] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None

class AirmetResponse(WeatherResponseBase):
    id: str
    raw_text: Optional[str] = None
    area: Optional[List[Dict[str, float]]] = None
    altitude: Optional[Dict[str, int]] = None
    phenomenon: Optional[str] = None
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None

class WeatherProductRequest(BaseModel):
    station: Optional[str] = None
    location: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    bbox: Optional[str] = None
    hours: Optional[int] = 1
