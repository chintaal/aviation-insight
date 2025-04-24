from typing import Dict, Any, List, Optional
import logging
import re
from datetime import datetime

from app.services.base_client import BaseApiClient
from app.schemas.weather import PirepResponse

logger = logging.getLogger(__name__)

class PirepService(BaseApiClient):
    """Client for NOAA Aviation Weather Center PIREP API"""
    
    def __init__(self):
        super().__init__(base_url="https://aviationweather.gov")
        
    async def get_pireps(self, station: str, distance: int = 200, age: float = 1.5) -> List[PirepResponse]:
        """
        Get PIREP data from Aviation Weather Center API using the simplified endpoint
        
        Args:
            station: ICAO station code (e.g., KATL)
            distance: Search radius in nm (default: 200)
            age: Age limit in hours (default: 1.5)
            
        Returns:
            List of PirepResponse objects
        """
        try:
            endpoint = "/api/data/pirep"
            params = {
                "id": station,
                "distance": distance,
                "age": age,
                "format": "raw"
            }
                
            # Specify response_type="text" since AWC returns plain text for PIREPs
            data = await self.get(endpoint, params=params, response_type="text")
            
            if not data or not isinstance(data, str) or not data.strip():
                return [PirepResponse(
                    source="AWC",
                    location=station,
                    raw_text="No PIREP data available"
                )]
            
            # Parse the raw CSV-like text response
            return self._parse_raw_pireps(data)
            
        except Exception as e:
            logger.error(f"Error fetching PIREPs: {str(e)}")
            return [PirepResponse(
                source="AWC",
                location=station,
                raw_text=f"Error fetching PIREPs: {str(e)}"
            )]
    
    def _parse_raw_pireps(self, raw_data: str) -> List[PirepResponse]:
        """Parse raw PIREP data from AWC into structured format"""
        results = []
        
        # Split the response into lines/individual PIREPs
        lines = raw_data.strip().split('\n')
        
        for line in lines:
            parts = line.split(' ', 1)  # Split into location and the report
            if len(parts) != 2:
                continue
                
            location = parts[0]
            raw_text = parts[1]
            
            # Create basic PIREP object
            pirep = PirepResponse(
                source="AWC",
                location=location,
                raw_text=raw_text
            )
            
            # Extract fields using regex patterns
            self._extract_pirep_fields(pirep, raw_text)
            
            results.append(pirep)
        
        return results
    
    def _extract_pirep_fields(self, pirep: PirepResponse, raw_text: str) -> None:
        """Extract fields from raw PIREP text"""
        # Report type (UA or UUA)
        if raw_text.startswith("UA "):
            pirep.report_type = "UA"  # Routine PIREP
        elif raw_text.startswith("UUA "):
            pirep.report_type = "UUA"  # Urgent PIREP
            
        # Extract aircraft type
        tp_match = re.search(r'/TP\s+([^/]+)', raw_text)
        if tp_match:
            pirep.aircraft_type = tp_match.group(1).strip()
            
        # Extract altitude
        fl_match = re.search(r'/FL(\d{3}|\d{2}|DUR(?:C|GD|G|D)?)', raw_text)
        if fl_match:
            fl_value = fl_match.group(1)
            if fl_value.isdigit():
                pirep.altitude = int(fl_value) * 100  # FL300 = 30,000 ft
            else:
                pirep.altitude = fl_value  # Could be DURGD, etc.
                
        # Extract time
        tm_match = re.search(r'/TM\s+(\d{4})', raw_text)
        if tm_match:
            time_str = tm_match.group(1)
            # Create timestamp from today's date and the time
            now = datetime.utcnow()
            hour = int(time_str[:2])
            minute = int(time_str[2:])
            try:
                pirep.timestamp = datetime(now.year, now.month, now.day, hour, minute).isoformat()
            except ValueError:
                pass  # Invalid time
                
        # Extract turbulence info
        tb_match = re.search(r'/TB\s+([^/]+)', raw_text)
        if tb_match:
            turbulence_text = tb_match.group(1).strip()
            
            if 'NEG' not in turbulence_text.upper():
                intensity = 'UNKNOWN'
                
                # Extract intensity
                if 'LGT' in turbulence_text.upper() and 'MOD' in turbulence_text.upper():
                    intensity = 'LGT-MOD'
                elif 'MOD' in turbulence_text.upper() and 'SEV' in turbulence_text.upper():
                    intensity = 'MOD-SEV'
                elif 'SEV' in turbulence_text.upper():
                    intensity = 'SEV'
                elif 'MOD' in turbulence_text.upper():
                    intensity = 'MOD'
                elif 'LGT' in turbulence_text.upper():
                    intensity = 'LGT'
                
                # Extract type/frequency
                frequency = None
                if 'CONS' in turbulence_text.upper():
                    frequency = 'CONS'
                elif 'INTMT' in turbulence_text.upper() or 'INTRMT' in turbulence_text.upper():
                    frequency = 'INTMT'
                elif 'OCNL' in turbulence_text.upper():
                    frequency = 'OCNL'
                    
                # Create turbulence dict
                turbulence_info = {
                    "intensity": intensity
                }
                
                if frequency:
                    turbulence_info["frequency"] = frequency
                    
                # Extract altitude if specified
                alt_match = re.search(r'\d{3}', tb_match.group(1))
                if alt_match:
                    turbulence_info["altitude"] = int(alt_match.group(0)) * 100
                    
                pirep.turbulence = turbulence_info
                
        # Extract icing info
        ic_match = re.search(r'/IC\s+([^/]+)', raw_text)
        if ic_match:
            icing_text = ic_match.group(1).strip()
            
            if 'NEG' not in icing_text.upper():
                intensity = 'UNKNOWN'
                
                # Extract intensity
                if 'LGT' in icing_text.upper() and 'MOD' in icing_text.upper():
                    intensity = 'LGT-MOD'
                elif 'MOD' in icing_text.upper() and 'SEV' in icing_text.upper():
                    intensity = 'MOD-SEV'
                elif 'SEV' in icing_text.upper():
                    intensity = 'SEV'
                elif 'MOD' in icing_text.upper():
                    intensity = 'MOD'
                elif 'LGT' in icing_text.upper():
                    intensity = 'LGT'
                elif 'TRC' in icing_text.upper():
                    intensity = 'TRACE'
                    
                # Extract type
                ice_type = None
                if 'RIME' in icing_text.upper():
                    ice_type = 'RIME'
                elif 'CLEAR' in icing_text.upper():
                    ice_type = 'CLEAR'
                elif 'MIXED' in icing_text.upper():
                    ice_type = 'MIXED'
                    
                # Create icing dict
                icing_info = {
                    "intensity": intensity
                }
                
                if ice_type:
                    icing_info["type"] = ice_type
                    
                pirep.icing = icing_info
                
        # Extract sky conditions
        sk_match = re.search(r'/SK\s+([^/]+)', raw_text)
        if sk_match:
            pirep.sky_conditions = sk_match.group(1).strip()
                
        # Extract remarks
        rm_match = re.search(r'/RM\s+(.+)$', raw_text)
        if rm_match:
            pirep.remarks = rm_match.group(1).strip()
