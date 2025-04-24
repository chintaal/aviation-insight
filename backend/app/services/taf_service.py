from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from app.services.base_client import BaseApiClient
from app.schemas.weather import TafResponse
from app.core.config import settings
from app.services.taf_parser import parse_taf

logger = logging.getLogger(__name__)

class AWCTafService(BaseApiClient):
    """Client for NOAA Aviation Weather Center TAF API"""
    
    def __init__(self):
        super().__init__(base_url="https://aviationweather.gov")
        
    async def get_taf(self, station: str, hours: int = 6) -> TafResponse:
        """Get TAF data from Aviation Weather Center API"""
        try:
            # Updated endpoint to match current AWC API structure
            endpoint = "/api/data/taf"
            params = {
                "ids": station,
                "hours": hours,
                "format": "json"
            }
            
            data = await self.get(endpoint, params=params)
            
            if not data or not isinstance(data, list) or len(data) == 0:
                return TafResponse(
                    source="AWC",
                    station=station,
                    raw_text=f"No TAF data available for {station}"
                )
            
            taf_data = data[0]
            raw_taf = taf_data.get("rawTAF", "")
            
            # Parse TAF using our enhanced parser
            parsed_taf = None
            pilot_summary = None
            
            # Get forecast periods from the AWC API response
            forecast_periods = []
            if "fcsts" in taf_data:
                forecast_periods = taf_data.get("fcsts", [])
            
            if raw_taf:
                try:
                    parsed_taf = parse_taf(raw_taf)
                    pilot_summary = parsed_taf.get("pilot_summary")
                    
                    # Enhanced forecast periods with detailed insights if available
                    if parsed_taf and parsed_taf.get("forecast_periods"):
                        forecast_periods = parsed_taf.get("forecast_periods")
                except Exception as e:
                    logger.error(f"Error parsing TAF: {str(e)}")
            
            # Parse timestamp fields safely with proper type checking
            issue_time = self._parse_timestamp(taf_data.get("issueTime"))
            valid_from = self._parse_timestamp(taf_data.get("validTimeFrom"))
            valid_to = self._parse_timestamp(taf_data.get("validTimeTo"))
            
            # Extract relevant fields from the response
            result = TafResponse(
                source="AWC",
                station=station,
                raw_text=raw_taf,
                issue_time=issue_time,
                valid_from=valid_from,
                valid_to=valid_to,
                forecast=forecast_periods,
                raw_data=taf_data,
                parsed_taf=parsed_taf,
                pilot_summary=pilot_summary
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching TAF from AWC: {str(e)}")
            return TafResponse(
                source="AWC",
                station=station,
                raw_text=f"Error fetching TAF: {str(e)}"
            )
            
    def _parse_timestamp(self, timestamp_value) -> Optional[datetime]:
        """Safely parse timestamp values that might be strings or integers"""
        if timestamp_value is None:
            return None
            
        if isinstance(timestamp_value, str):
            try:
                # Remove Z and replace with proper UTC offset for ISO format
                return datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
            except (ValueError, AttributeError) as e:
                logger.error(f"Error parsing string timestamp: {str(e)}")
                return None
                
        elif isinstance(timestamp_value, int):
            try:
                # Handle both seconds and milliseconds timestamp formats
                from datetime import timezone
                if timestamp_value > 1000000000000:  # Likely milliseconds
                    return datetime.fromtimestamp(timestamp_value / 1000, tz=timezone.utc)
                else:  # Likely seconds
                    return datetime.fromtimestamp(timestamp_value, tz=timezone.utc)
            except (ValueError, OverflowError) as e:
                logger.error(f"Error parsing integer timestamp {timestamp_value}: {str(e)}")
                return None
        
        # If it's neither a string nor an integer, log and return None
        logger.error(f"Unexpected timestamp type: {type(timestamp_value)}")
        return None

class AVWXTafService(BaseApiClient):
    """Client for AVWX TAF API"""
    
    def __init__(self):
        # Use the hardcoded API key
        api_key = "  "
        super().__init__(
            base_url="https://avwx.rest/api", 
            api_key=api_key
        )
        
    async def get_taf(self, station: str) -> TafResponse:
        """Get TAF data from AVWX API"""
        try:
            endpoint = f"/taf/{station}"
            
            # Always use the hardcoded API key for authorization
            headers = {"Authorization": "  "}
            data = await self.get(endpoint, headers=headers)
            
            # Extract relevant fields from the response
            raw_taf = data.get("raw")
            
            # Use the improved parse_timestamp method from AWCTafService
            # We'll create a helper method for consistency
            issue_time = self._parse_timestamp(data.get("time", {}).get("dt") if isinstance(data.get("time"), dict) else None)
            valid_from = self._parse_timestamp(data.get("start_time"))
            valid_to = self._parse_timestamp(data.get("end_time"))
            
            # Parse TAF using our enhanced parser
            parsed_taf = None
            pilot_summary = None
            forecast_periods = data.get("forecast", [])
            
            if raw_taf:
                try:
                    parsed_taf = parse_taf(raw_taf)
                    pilot_summary = parsed_taf.get("pilot_summary")
                    
                    # If we have our own detailed forecast periods, add them
                    # But keep AVWX periods as well for compatibility
                    if parsed_taf and parsed_taf.get("forecast_periods"):
                        # For each period from our parser, add a "pilot_insights" key
                        for period in parsed_taf.get("forecast_periods", []):
                            if period.get("pilot_summary"):
                                # Find matching period in AVWX data if possible
                                for avwx_period in forecast_periods:
                                    # If we have matching times, add our insights
                                    if self._periods_overlap(period, avwx_period):
                                        avwx_period["pilot_insights"] = period.get("pilot_summary")
                                        avwx_period["flight_rules_details"] = period.get("flight_rules")
                                        if period.get("planning_considerations"):
                                            avwx_period["planning_considerations"] = period.get("planning_considerations")
                                        break
                except Exception as e:
                    logger.error(f"Error parsing TAF: {str(e)}")
            
            result = TafResponse(
                source="AVWX",
                station=station,
                raw_text=raw_taf,
                issue_time=issue_time,
                valid_from=valid_from,
                valid_to=valid_to,
                forecast=forecast_periods,
                raw_data=data,
                parsed_taf=parsed_taf,
                pilot_summary=pilot_summary
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching TAF from AVWX: {str(e)}")
            return TafResponse(
                source="AVWX",
                station=station,
                raw_text=f"Error fetching TAF: {str(e)}"
            )
    
    def _periods_overlap(self, period1: Dict[str, Any], period2: Dict[str, Any]) -> bool:
        """Check if two forecast periods overlap in time"""
        # Try to get datetime objects from both periods
        p1_start = period1.get("valid_from")
        p1_end = period1.get("valid_to")
        
        # For AVWX format
        p2_start = None
        p2_end = None
        
        if period2.get("start_time") and isinstance(period2["start_time"], dict) and period2["start_time"].get("dt"):
            try:
                p2_start = datetime.fromisoformat(period2["start_time"]["dt"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass
                
        if period2.get("end_time") and isinstance(period2["end_time"], dict) and period2["end_time"].get("dt"):
            try:
                p2_end = datetime.fromisoformat(period2["end_time"]["dt"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass
        
        # If we can't get proper time objects, assume they match
        if not all([p1_start, p1_end, p2_start, p2_end]):
            return True
            
        # Check for overlap
        return (p1_start <= p2_end) and (p2_start <= p1_end)
    
    def _parse_timestamp(self, timestamp_value) -> Optional[datetime]:
        """Safely parse timestamp values that might be strings or integers"""
        if timestamp_value is None:
            return None
            
        if isinstance(timestamp_value, str):
            try:
                # Remove Z and replace with proper UTC offset for ISO format
                return datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
            except (ValueError, AttributeError) as e:
                logger.error(f"Error parsing string timestamp: {str(e)}")
                return None
                
        elif isinstance(timestamp_value, int):
            try:
                # Handle both seconds and milliseconds timestamp formats
                from datetime import timezone
                if timestamp_value > 1000000000000:  # Likely milliseconds
                    return datetime.fromtimestamp(timestamp_value / 1000, tz=timezone.utc)
                else:  # Likely seconds
                    return datetime.fromtimestamp(timestamp_value, tz=timezone.utc)
            except (ValueError, OverflowError) as e:
                logger.error(f"Error parsing integer timestamp {timestamp_value}: {str(e)}")
                return None
        
        # If it's neither a string nor an integer, log and return None
        logger.error(f"Unexpected timestamp type: {type(timestamp_value)}")
        return None
