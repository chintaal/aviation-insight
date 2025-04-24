from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import re

from app.services.base_client import BaseApiClient
from app.services.metar_parser import parse_metar
from app.schemas.weather import MetarResponse

logger = logging.getLogger(__name__)

class AWCMetarService(BaseApiClient):
    """Client for NOAA Aviation Weather Center METAR API"""
    
    def __init__(self):
        super().__init__(base_url="https://aviationweather.gov")
        
    async def get_metar(self, station: str, hours: int = 1) -> MetarResponse:
        """Get METAR data from Aviation Weather Center API"""
        try:
            # Use the correct endpoint for AWC METAR API
            endpoint = "/api/data/metar"
            params = {
                "ids": station,
                "format": "json",
                "hours": hours
            }
            
            data = await self.get(endpoint, params=params)
            
            if not data or len(data) == 0:
                return MetarResponse(
                    source="AWC",
                    station=station,
                    raw_text=f"No METAR data available for {station}"
                )
            
            metar_data = data[0]
            raw_metar = metar_data.get("rawOb")
            
            # Process the raw METAR through our parser to get detailed information and pilot summary
            parsed_data = {}
            if raw_metar:
                parsed_data = parse_metar(raw_metar)
            
            # Process visibility - handle special cases like "10+" by removing non-numeric characters
            visibility = metar_data.get("visib")
            if visibility is not None and not isinstance(visibility, (int, float)):
                # Extract numeric part if it's a string
                if isinstance(visibility, str):
                    # Remove any non-numeric characters except decimal point
                    visibility_str = re.sub(r'[^\d.]', '', visibility)
                    try:
                        visibility = float(visibility_str) if visibility_str else None
                    except ValueError:
                        visibility = None
            
            # Process wind direction - handle special cases like "VRB" (variable)
            wind_direction = metar_data.get("wdir")
            if wind_direction is not None and not isinstance(wind_direction, (int, float)):
                if isinstance(wind_direction, str):
                    if wind_direction.upper() == "VRB" or not wind_direction.isdigit():
                        wind_direction = None
                    else:
                        try:
                            wind_direction = int(wind_direction)
                        except ValueError:
                            wind_direction = None
            
            # Extract relevant fields from the response
            result = MetarResponse(
                source="AWC",
                station=station,
                raw_text=raw_metar,
                flight_category=metar_data.get("flightCategory") or parsed_data.get("flight_category"),
                temperature=metar_data.get("temp") or parsed_data.get("temperature"),
                dewpoint=metar_data.get("dewp") or parsed_data.get("dewpoint"),
                wind_speed=metar_data.get("wspd"),
                wind_direction=wind_direction,
                visibility=visibility,
                parsed_metar=parsed_data,  # Include our detailed parsed data
                pilot_summary=parsed_data.get("pilot_summary", ""),  # Include the pilot-friendly summary
                raw_data=metar_data
            )
            
            # Extract ceiling information if available
            if "clouds" in metar_data and metar_data["clouds"]:
                result.clouds = metar_data["clouds"]
                ceiling = None
                for cloud in metar_data["clouds"]:
                    if cloud.get("cover") in ["BKN", "OVC"]:
                        ceiling = cloud.get("base")
                        break
                result.ceiling = ceiling or parsed_data.get("ceiling")
            elif parsed_data.get("clouds"):
                result.clouds = parsed_data.get("clouds")
                result.ceiling = parsed_data.get("ceiling")
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching METAR from AWC: {str(e)}")
            return MetarResponse(
                source="AWC",
                station=station,
                raw_text=f"Error fetching METAR: {str(e)}"
            )
