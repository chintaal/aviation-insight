from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from app.services.base_client import BaseApiClient
from app.schemas.weather import SigmetResponse, AirmetResponse
from app.core.config import settings

logger = logging.getLogger(__name__)

class AWCSigmetService(BaseApiClient):
    """Client for NOAA Aviation Weather Center SIGMET API"""
    
    def __init__(self):
        super().__init__(base_url="https://aviationweather.gov")
        
    async def get_sigmets(self, region: str = "all") -> List[SigmetResponse]:
        """Get SIGMET data from Aviation Weather Center API"""
        try:
            endpoint = "/data/api/airsigmet"
            params = {
                "region": region,
                "format": "json"
            }
            
            data = await self.get(endpoint, params=params)
            
            if "data" not in data or not data["data"]:
                return [SigmetResponse(
                    source="AWC",
                    id=f"no-sigmets-{region}",
                    raw_text=f"No SIGMET data available for region {region}"
                )]
            
            results = []
            for sigmet_data in data["data"]:
                # Only include SIGMET (not AIRMET)
                if sigmet_data.get("airsigmetType", "").lower() != "sigmet":
                    continue
                
                # Extract relevant fields from the response
                result = SigmetResponse(
                    source="AWC",
                    id=sigmet_data.get("airsigmetId", "unknown"),
                    raw_text=sigmet_data.get("rawAirSigmet"),
                    phenomenon=sigmet_data.get("hazard"),
                    valid_from=datetime.fromisoformat(sigmet_data.get("validTimeFrom", "").replace('Z', '+00:00')) if sigmet_data.get("validTimeFrom") else None,
                    valid_to=datetime.fromisoformat(sigmet_data.get("validTimeTo", "").replace('Z', '+00:00')) if sigmet_data.get("validTimeTo") else None,
                    raw_data=sigmet_data
                )
                
                # Extract area geometry if available
                if "geometry" in sigmet_data:
                    geometry = sigmet_data["geometry"]
                    if geometry.get("type") == "Polygon" and "coordinates" in geometry:
                        coords = geometry["coordinates"][0]  # First polygon
                        result.area = [{"lat": coord[1], "lon": coord[0]} for coord in coords]
                
                # Extract altitude information if available
                if "altitudeLower" in sigmet_data or "altitudeUpper" in sigmet_data:
                    result.altitude = {}
                    if "altitudeLower" in sigmet_data:
                        result.altitude["lower"] = sigmet_data["altitudeLower"]
                    if "altitudeUpper" in sigmet_data:
                        result.altitude["upper"] = sigmet_data["altitudeUpper"]
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching SIGMET from AWC: {str(e)}")
            return [SigmetResponse(
                source="AWC",
                id="error",
                raw_text=f"Error fetching SIGMET: {str(e)}"
            )]

class AWCAirmetService(BaseApiClient):
    """Client for NOAA Aviation Weather Center AIRMET API"""
    
    def __init__(self):
        super().__init__(base_url="https://aviationweather.gov")
        
    async def get_airmets(self, region: str = "all") -> List[AirmetResponse]:
        """Get AIRMET data from Aviation Weather Center API"""
        try:
            endpoint = "/data/api/airsigmet"
            params = {
                "region": region,
                "format": "json"
            }
            
            data = await self.get(endpoint, params=params)
            
            if "data" not in data or not data["data"]:
                return [AirmetResponse(
                    source="AWC",
                    id=f"no-airmets-{region}",
                    raw_text=f"No AIRMET data available for region {region}"
                )]
            
            results = []
            for airmet_data in data["data"]:
                # Only include AIRMET (not SIGMET)
                if airmet_data.get("airsigmetType", "").lower() != "airmet":
                    continue
                
                # Extract relevant fields from the response
                result = AirmetResponse(
                    source="AWC",
                    id=airmet_data.get("airsigmetId", "unknown"),
                    raw_text=airmet_data.get("rawAirSigmet"),
                    phenomenon=airmet_data.get("hazard"),
                    valid_from=datetime.fromisoformat(airmet_data.get("validTimeFrom", "").replace('Z', '+00:00')) if airmet_data.get("validTimeFrom") else None,
                    valid_to=datetime.fromisoformat(airmet_data.get("validTimeTo", "").replace('Z', '+00:00')) if airmet_data.get("validTimeTo") else None,
                    raw_data=airmet_data
                )
                
                # Extract area geometry if available
                if "geometry" in airmet_data:
                    geometry = airmet_data["geometry"]
                    if geometry.get("type") == "Polygon" and "coordinates" in geometry:
                        coords = geometry["coordinates"][0]  # First polygon
                        result.area = [{"lat": coord[1], "lon": coord[0]} for coord in coords]
                
                # Extract altitude information if available
                if "altitudeLower" in airmet_data or "altitudeUpper" in airmet_data:
                    result.altitude = {}
                    if "altitudeLower" in airmet_data:
                        result.altitude["lower"] = airmet_data["altitudeLower"]
                    if "altitudeUpper" in airmet_data:
                        result.altitude["upper"] = airmet_data["altitudeUpper"]
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching AIRMET from AWC: {str(e)}")
            return [AirmetResponse(
                source="AWC",
                id="error",
                raw_text=f"Error fetching AIRMET: {str(e)}"
            )]

class AVWXSigmetService(BaseApiClient):
    """Client for AVWX SIGMET API"""
    
    def __init__(self):
        super().__init__(
            base_url="https://avwx.rest/api", 
            api_key=settings.AVWX_API_KEY
        )
        
    async def get_sigmets(self) -> List[SigmetResponse]:
        """Get SIGMET data from AVWX API"""
        try:
            if not settings.AVWX_API_KEY:
                return [SigmetResponse(
                    source="AVWX",
                    id="no-key",
                    raw_text="No API key provided for AVWX"
                )]
            
            endpoint = "/sigmets"
            
            headers = {"Authorization": settings.AVWX_API_KEY}
            data = await self.get(endpoint, headers=headers)
            
            if not data:
                return [SigmetResponse(
                    source="AVWX",
                    id="no-data",
                    raw_text="No SIGMET data available"
                )]
            
            results = []
            for sigmet_data in data:
                # Extract relevant fields from the response
                result = SigmetResponse(
                    source="AVWX",
                    id=sigmet_data.get("id", "unknown"),
                    raw_text=sigmet_data.get("raw"),
                    phenomenon=sigmet_data.get("hazard"),
                    valid_from=datetime.fromisoformat(sigmet_data.get("start_time", "").replace('Z', '+00:00')) if sigmet_data.get("start_time") else None,
                    valid_to=datetime.fromisoformat(sigmet_data.get("end_time", "").replace('Z', '+00:00')) if sigmet_data.get("end_time") else None,
                    raw_data=sigmet_data
                )
                
                # Extract area geometry if available
                if "geojson" in sigmet_data and sigmet_data["geojson"]:
                    if "coordinates" in sigmet_data["geojson"]:
                        coords = sigmet_data["geojson"]["coordinates"]
                        result.area = [{"lat": coord[1], "lon": coord[0]} for coord in coords]
                
                # Extract altitude information if available
                if "altitude" in sigmet_data and sigmet_data["altitude"]:
                    result.altitude = {}
                    if "min" in sigmet_data["altitude"]:
                        result.altitude["lower"] = sigmet_data["altitude"]["min"]["value"]
                    if "max" in sigmet_data["altitude"]:
                        result.altitude["upper"] = sigmet_data["altitude"]["max"]["value"]
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching SIGMET from AVWX: {str(e)}")
            return [SigmetResponse(
                source="AVWX",
                id="error",
                raw_text=f"Error fetching SIGMET: {str(e)}"
            )]

class AVWXAirmetService(BaseApiClient):
    """Client for AVWX AIRMET API"""
    
    def __init__(self):
        super().__init__(
            base_url="https://avwx.rest/api", 
            api_key=settings.AVWX_API_KEY
        )
        
    async def get_airmets(self) -> List[AirmetResponse]:
        """Get AIRMET data from AVWX API"""
        try:
            if not settings.AVWX_API_KEY:
                return [AirmetResponse(
                    source="AVWX",
                    id="no-key",
                    raw_text="No API key provided for AVWX"
                )]
            
            endpoint = "/airmets"
            
            headers = {"Authorization": settings.AVWX_API_KEY}
            data = await self.get(endpoint, headers=headers)
            
            if not data:
                return [AirmetResponse(
                    source="AVWX",
                    id="no-data",
                    raw_text="No AIRMET data available"
                )]
            
            results = []
            for airmet_data in data:
                # Extract relevant fields from the response
                result = AirmetResponse(
                    source="AVWX",
                    id=airmet_data.get("id", "unknown"),
                    raw_text=airmet_data.get("raw"),
                    phenomenon=airmet_data.get("hazard"),
                    valid_from=datetime.fromisoformat(airmet_data.get("start_time", "").replace('Z', '+00:00')) if airmet_data.get("start_time") else None,
                    valid_to=datetime.fromisoformat(airmet_data.get("end_time", "").replace('Z', '+00:00')) if airmet_data.get("end_time") else None,
                    raw_data=airmet_data
                )
                
                # Extract area geometry if available
                if "geojson" in airmet_data and airmet_data["geojson"]:
                    if "coordinates" in airmet_data["geojson"]:
                        coords = airmet_data["geojson"]["coordinates"]
                        result.area = [{"lat": coord[1], "lon": coord[0]} for coord in coords]
                
                # Extract altitude information if available
                if "altitude" in airmet_data and airmet_data["altitude"]:
                    result.altitude = {}
                    if "min" in airmet_data["altitude"]:
                        result.altitude["lower"] = airmet_data["altitude"]["min"]["value"]
                    if "max" in airmet_data["altitude"]:
                        result.altitude["upper"] = airmet_data["altitude"]["max"]["value"]
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error fetching AIRMET from AVWX: {str(e)}")
            return [AirmetResponse(
                source="AVWX",
                id="error",
                raw_text=f"Error fetching AIRMET: {str(e)}"
            )]
